"""Helpers for CT1 (integral shape descriptor), CT2 (local spatial consistency) and CT3 (error diagnosis)."""
import math
import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

EARTH_R = 6378137.0  # same radius as evaluateDistance.latlog2meter


def load_gps(root):
    """class folder name -> (lon, lat), parsed the same way as evaluateDistance.py."""
    gps = {}
    for line in open(f'{root}/Dense_GPS_ALL.txt'):
        p = line.split(' ')
        gps[p[0].split('/')[-2]] = (float(p[1].split('E')[-1]), float(p[2].split('N')[-1]))
    return gps


def class_xy(names, gps):
    """Class positions in metres around the mean of all GPS points (equirectangular projection:
    about 0.5% scale error 50 km from the reference, negligible for the 20-60 m thresholds used here)."""
    lon0, lat0 = np.mean(list(gps.values()), 0)
    ll = np.array([gps[n] for n in names], dtype=np.float64)
    x = np.radians(ll[:, 0] - lon0) * math.cos(math.radians(lat0)) * EARTH_R
    y = np.radians(ll[:, 1] - lat0) * EARTH_R
    return np.stack([x, y], 1)


# ---------------- CT2: local spatial consistency ----------------

class GeoSampler(torch.utils.data.Sampler):
    """Random permutation of the classes in which every class is followed by up to group-1 unused
    neighbours within `radius` m. Each class still appears once per epoch (as Sampler_University with
    sample_num=1), so labels inside a batch stay distinct, which the repo's triplet mining needs."""

    def __init__(self, xy, group=4, radius=30.0):
        tree = cKDTree(xy)
        self.nbrs = [[j for j in tree.query_ball_point(p, radius) if j != i] for i, p in enumerate(xy)]
        self.group = group

    def __iter__(self):
        used = np.zeros(len(self.nbrs), bool)
        order = []
        for a in np.random.permutation(len(self.nbrs)):
            if used[a]:
                continue
            free = [j for j in self.nbrs[a] if not used[j]]
            np.random.shuffle(free)
            g = [a] + free[:self.group - 1]
            used[g] = True
            order += g
        return iter(order)

    def __len__(self):
        return len(self.nbrs)


def local_consistency_loss(feat, xy, near=30.0, far=60.0, alpha=3.0):
    """Ordinal local consistency: for every sample that has a <= `near` m neighbour in the batch, its
    hardest neighbour must be closer in feature space than its closest place > `far` m away.
    No Lipschitz constant is imposed, only the order near < far. Same soft form and alpha as the
    repo's WeightedSoftTripletLoss. feat: (n, d) features of both views, xy: (n, 2) metres."""
    f = F.normalize(feat.float(), dim=1)
    d_f = (2 - 2 * f @ f.T).clamp_min(1e-12).sqrt()
    d_g = torch.cdist(xy, xy)
    is_near = (d_g > 0) & (d_g <= near)
    is_far = d_g > far
    ok = is_near.any(1) & is_far.any(1)
    if not ok.any():
        return d_f.new_zeros(())
    d_near = d_f.masked_fill(~is_near, float('-inf')).amax(1)[ok]
    d_far = d_f.masked_fill(~is_far, float('inf')).amin(1)[ok]
    return F.softplus(alpha * (d_near - d_far)).mean()


# ---------------- CT1: integral geometric-semantic descriptor ----------------

def cell_kernels(n_r=4, n_t=8, s=112):
    """(n_r, n_t, s, s) Riemann-sum weights of  ∫_{U_i} f(y) dy  over the unit disc inscribed in the image,
    U_i = ring k x sector j as in Sec. 4.2 of the Lebesgue paper, f(y) = max(0, 1 - |y|) (f_decay, R = 1)."""
    u = (torch.arange(s) + 0.5) / s * 2 - 1
    y, x = torch.meshgrid(u, u, indexing='ij')
    r = (x ** 2 + y ** 2).sqrt()
    t = torch.atan2(y, x) % (2 * math.pi)
    ring = (r * n_r).long().clamp_max(n_r - 1)
    sec = (t / (2 * math.pi) * n_t).long().clamp_max(n_t - 1)
    w = (1 - r).clamp_min(0) * (2 / s) ** 2  # f_decay * pixel area, 0 outside the disc
    k = torch.zeros(n_r, n_t, s, s)
    k[ring, sec, torch.arange(s)[:, None], torch.arange(s)[None]] = w
    return k


def integral_descriptor(tokens, centers, kernels, tau=0.05):
    """Shape vector of the Lebesgue paper with semantic classes taken from ViT patch tokens.
    tokens: (B, 1 + h*w, d) backbone output, centers: (C, d) unit-norm codebook, kernels: cell_kernels().
    m_c(y) = soft membership of y in class c (chi_{S_c} for tau -> 0), W[c, k, j] = ∫_{U_kj} f(y) m_c(y) dy,
    then |rFFT| over the sectors: a rotation shifts the sectors cyclically and a flip reverses them,
    neither changes the magnitude, so the descriptor ignores the unknown UAV heading."""
    p = F.normalize(tokens[:, 1:].float(), dim=-1)
    b, n, _ = p.shape
    h = int(round(n ** 0.5))
    m = torch.softmax(p @ centers.T / tau, -1).transpose(1, 2).reshape(b, -1, h, h)
    m = F.interpolate(m, size=kernels.shape[-1], mode='bilinear', align_corners=False)
    w = torch.einsum('bcxy,rtxy->bcrt', m, kernels)
    return torch.fft.rfft(w, dim=-1).abs().flatten(1)


def rerank(top, qf, gf, qw, gw, lam, k=50):
    """Re-order the first k candidates of every query by z(d_F) + lam * z(d_W); z = per-query z-score.
    d_F = 1 - cosine of the repo's features, d_W = L1 distance of integral descriptors."""
    cand = top[:, :k]
    d_f = 1 - np.einsum('qd,qkd->qk', qf, gf[cand])
    d_w = np.abs(qw[:, None] - gw[cand]).sum(-1)
    z = lambda a: (a - a.mean(1, keepdims=True)) / (a.std(1, keepdims=True) + 1e-12)
    return np.take_along_axis(cand, np.argsort(z(d_f) + lam * z(d_w), 1, kind='stable'), 1)


# ---------------- metrics (same definitions as evaluate_gpu.py / evaluateDistance.py) ----------------

def recall(top, q_lab, g_lab, k):
    return 100 * (g_lab[top[:, :k]] == q_lab[:, None]).any(1).mean()


def sdm(top, q_ll, g_ll, k):
    """SDM@K as evaluateDistance.py: Euclidean distance in degrees, exp(-5e3 d), weights (K - i) / K."""
    d = np.linalg.norm(g_ll[top[:, :k]] - q_ll[:, None], axis=-1)
    w = (k - np.arange(k)) / k
    return 100 * ((np.exp(-5e3 * d) * w).sum(1) / w.sum()).mean()


# ---------------- CT3: error diagnosis ----------------

def areas(xy, gap=200.0):
    """Connected components of the sampling points with links <= gap m (one per flight area / campus)."""
    pairs = cKDTree(xy).query_pairs(gap, output_type='ndarray')
    adj = coo_matrix((np.ones(len(pairs)), pairs.T), shape=(len(xy), len(xy)))
    return connected_components(adj, directed=False)[1]


def error_types(pred, gt, xy, area, is_test, near=30.0):
    """Per query: C0 correct, C1 wrong cell <= near m away, C2 same area > near m, C3 another test area,
    C4 a training location (train and test universities are disjoint, so C4 is always inter-campus)."""
    d = np.linalg.norm(xy[pred] - xy[gt], axis=1)
    cat = np.select([pred == gt, d <= near, ~is_test[pred], area[pred] == area[gt]],
                    ['C0', 'C1', 'C4', 'C2'], 'C3')
    return cat, d


def local_ratio(d_f, xy, near=30.0):
    """Local Lipschitz ratio rho(i) = max_j d_F(i, j) / d_geo(i, j) over neighbours 0 < d_geo <= near (0 if none)."""
    d_g = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
    m = (d_g > 0) & (d_g <= near)
    return np.where(m, d_f / np.where(m, d_g, 1), 0).max(1)


def _selftest():
    rng = np.random.default_rng(0)
    # 20 m grid of two 10x10 areas 5 km apart
    g = np.stack(np.meshgrid(np.arange(10), np.arange(10)), -1).reshape(-1, 2) * 20.0
    xy = np.concatenate([g, g + 5000])
    # sampler: a permutation, distinct labels per batch, neighbours mostly together
    np.random.seed(0)
    order = list(GeoSampler(xy))
    assert sorted(order) == list(range(len(xy)))
    def pairs_per_batch(o):
        b = xy[np.array(o[:len(o) // 16 * 16])].reshape(-1, 16, 2)
        d = np.linalg.norm(b[:, :, None] - b[:, None], axis=-1)
        return ((d > 0) & (d <= 30)).sum() / 2 / len(b)
    geo, rnd = pairs_per_batch(order), pairs_per_batch(list(np.random.permutation(len(xy))))
    assert geo > 3 * rnd, (geo, rnd)
    # loss: ~0 when neighbours are close and far places far, large when reversed
    p = torch.tensor(xy[[0, 1, 150]], dtype=torch.float32)  # 0-1 are 20 m apart, 150 is 5 km away
    good = torch.tensor([[1., 0], [0.9, 0.1], [0, 1]])
    bad = torch.tensor([[1., 0], [0, 1], [0.9, 0.1]])
    assert local_consistency_loss(good, p) < 0.2 < 1.0 < local_consistency_loss(bad, p)
    # descriptor: invariant to 90 degree rotation and flip of the patch grid, sensitive to layout
    k = cell_kernels()
    centers = F.normalize(torch.randn(8, 16), dim=1)
    t = torch.randn(1, 1 + 14 * 14, 16)
    grid = t[:, 1:].reshape(1, 14, 14, 16)
    w0 = integral_descriptor(t, centers, k)
    for g2 in (grid.rot90(1, (1, 2)), grid.flip(2)):
        w1 = integral_descriptor(torch.cat([t[:, :1], g2.reshape(1, -1, 16)], 1), centers, k)
        assert (w1 - w0).abs().sum() / w0.abs().sum() < 0.02
    shuffled = torch.cat([t[:, :1], t[:, 1 + torch.randperm(196)]], 1)
    assert (integral_descriptor(shuffled, centers, k) - w0).abs().sum() / w0.abs().sum() > 0.1
    assert abs(k.sum().item() - math.pi / 3) < 0.01  # ∫_disc (1 - |y|) dy = pi / 3
    # rerank with lam = 0 keeps the cosine order; metrics on a toy ranking
    qf = F.normalize(torch.randn(5, 8), dim=1).numpy()
    gf = F.normalize(torch.randn(30, 8), dim=1).numpy()
    top = np.argsort(-(qf @ gf.T), 1)
    assert (rerank(top, qf, gf, rng.random((5, 3)), rng.random((30, 3)), 0.0, k=10) == top[:, :10]).all()
    lab = np.arange(30)
    assert recall(top, top[:, 0], lab, 1) == 100 and recall(top, top[:, 3], lab, 3) == 0
    ll = rng.random((30, 2))
    assert abs(sdm(top, ll[top[:, 0]], ll, 1) - 100) < 1e-9
    # diagnosis categories
    area = areas(xy)
    assert area.max() == 1
    is_test = np.arange(len(xy)) < 100  # first area = test, second = train
    cat, _ = error_types(np.array([0, 1, 50, 150]), np.array([0, 0, 0, 0]), xy, area, is_test)
    assert list(cat) == ['C0', 'C1', 'C2', 'C4'], cat
    d_f = np.linalg.norm(xy[:, None] - xy[None], axis=-1) / 100  # features vary 0.01 per metre
    assert np.allclose(local_ratio(d_f, xy), 0.01)
    print('geo_utils self-test passed')


if __name__ == '__main__':
    _selftest()

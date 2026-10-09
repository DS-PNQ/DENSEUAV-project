"""CT1 (integral descriptor re-ranking) and CT3 (error diagnosis) on the runs of the `ct` experiment.
Run after `python train_kaggle.py ct`. Usage: python analyze_ct.py   (tables and figures go to results/)"""
import json, os, re, sys
import matplotlib
matplotlib.use('Agg')
import pandas as pd
from configs import EXPERIMENTS, REPO_DIR, RESULTS_DIR, data_root

OUT = RESULTS_DIR
os.makedirs(OUT, exist_ok=True)
ROOT = data_root()
RUNS = [r['name'] for r in EXPERIMENTS['ct']]
os.chdir(REPO_DIR)  # checkpoints/ paths and the repo's models package
sys.path.insert(0, REPO_DIR)
display = lambda df: print(df.to_string(), end='\n\n')

import yaml, scipy.io
import numpy as np, torch, torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from PIL import Image
from types import SimpleNamespace
from torchvision import datasets as tvd, transforms
from sklearn.cluster import KMeans
from geo_utils import load_gps, class_xy, cell_kernels, integral_descriptor, rerank, recall, sdm, areas, error_types, local_ratio
from models.taskflow import make_model

DONE = [n for n in RUNS if os.path.exists(f'checkpoints/{n}/pytorch_result_1.mat')]
tf = transforms.Compose([transforms.Resize((224, 224), interpolation=3), transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])  # test.py's transform
gal = tvd.ImageFolder(f'{ROOT}/test/gallery_satellite', tf)
qry = tvd.ImageFolder(f'{ROOT}/test/query_drone', tf)
names = np.array(gal.classes)
g_lab = np.array(gal.targets)                                              # gallery image -> place index
q_lab = np.array([gal.class_to_idx[qry.classes[t]] for t in qry.targets])  # query image -> place index
gps = load_gps(ROOT)
xy = class_xy(names, gps)                                                  # metres
ll = np.array([gps[n] for n in names])                                     # (lon, lat), for SDM
is_test = np.isin(np.arange(len(names)), q_lab)
area = areas(xy)
test_areas = np.unique(area[is_test])
print(f'{len(names)} places, {len(g_lab)} gallery / {len(q_lab)} query images, {area.max() + 1} areas, '
      f'{len(test_areas)} with queries (paper: 14 universities, 4 for test)')
print('areas mixing train and test places:', [a for a in test_areas if (~is_test[area == a]).any()])


def load_feats(name):
    m = scipy.io.loadmat(f'checkpoints/{name}/pytorch_result_1.mat')
    cls = np.array([int(n) for n in names])  # test.py's labels are int(folder name)
    assert (m['gallery_label'][0] == cls[g_lab]).all() and (m['query_label'][0] == cls[q_lab]).all()
    return m['query_f'].astype(np.float32), m['gallery_f'].astype(np.float32)


@torch.no_grad()
def topk(qf, gf, k=100):
    g = torch.from_numpy(gf).cuda()
    return torch.cat([(torch.from_numpy(qf[i:i + 512]).cuda() @ g.T).topk(k).indices.cpu()
                      for i in range(0, len(qf), 512)]).numpy()


def metrics(top):
    return {**{f'R@{k}': recall(top, q_lab, g_lab, k) for k in (1, 5, 10)},
            **{f'SDM@{k}': sdm(top, ll[q_lab], ll[g_lab], k) for k in (1, 3, 5, 10)}}


FEAT = {n: load_feats(n) for n in DONE}
TOP = {n: topk(*FEAT[n]) for n in DONE}
for n in DONE:
    r = dict(re.findall(r'([\w@]+):([\d.]+)', open(f'checkpoints/{n}/results.txt').read()))
    s = json.load(open(f'checkpoints/{n}/SDM@K(1,100).json'))
    mine = metrics(TOP[n])
    assert abs(mine['R@1'] - float(r['Recall@1'])) < 0.1 and abs(mine['SDM@1'] - 100 * s['1']) < 0.1, (n, mine)
print('R@1 and SDM@1 match the repo for', DONE)

N_R, N_T, C, LAMBDA, K_CAND = 4, 8, 8, 1.0, 50
KERN = cell_kernels(N_R, N_T).cuda()
rng = np.random.default_rng(0)
CODEBOOK_SET = torch.utils.data.ConcatDataset(  # 600 training images per view
    [torch.utils.data.Subset(d, rng.choice(len(d), min(600, len(d)), replace=False))
     for d in (tvd.ImageFolder(f'{ROOT}/train/{v}', tf) for v in ('drone', 'satellite'))])


def load_model(name):
    opt = SimpleNamespace(**yaml.safe_load(open(f'checkpoints/{name}/opts.yaml')))
    model = make_model(opt)
    model.load_state_dict(torch.load(f'checkpoints/{name}/net_119.pth', map_location='cpu'))
    return model.cuda().eval()


@torch.no_grad()
def over_tokens(model, ds, fn, bs=128):
    out = []
    for x, _ in torch.utils.data.DataLoader(ds, bs, num_workers=os.cpu_count()):
        with torch.autocast('cuda'):
            t = model.backbone(x.cuda())  # (B, 1 + 14*14, 384)
        out.append(fn(t.float()).cpu())
    return torch.cat(out)


DESC = {}
for n in DONE:
    model = load_model(n)
    tok = over_tokens(model, CODEBOOK_SET, lambda t: F.normalize(t[:, 1:], dim=-1)[:, ::4].flatten(0, 1))
    km = KMeans(C, n_init=4, random_state=0).fit(tok.numpy())
    centers = F.normalize(torch.tensor(km.cluster_centers_, dtype=torch.float32), dim=1).cuda()
    desc = lambda t: integral_descriptor(t, centers, KERN)
    DESC[n] = over_tokens(model, qry, desc).numpy(), over_tokens(model, gal, desc).numpy()
    del model
    torch.cuda.empty_cache()

RANK = {}
for n in DONE:
    RANK[n] = TOP[n]
    RANK[f'{n}+CT1'] = rerank(TOP[n], *FEAT[n], *DESC[n], LAMBDA, K_CAND)
res = pd.DataFrame({v: metrics(t) for v, t in RANK.items()}).T
res.loc['paper Table VI Global'] = [80.18, 93.99, np.nan, 84.39, 82.51, 78.02, 66.46]
res.round(2).to_csv(f'{OUT}/metrics.csv')
print(f'CT1 ceiling, a correct image in the top {K_CAND}:', {n: round(recall(TOP[n], q_lab, g_lab, K_CAND), 2) for n in DONE})
display(res.round(2))
sweep = pd.DataFrame({lam: metrics(rerank(TOP[DONE[0]], *FEAT[DONE[0]], *DESC[DONE[0]], lam, K_CAND))
                      for lam in (0, 0.25, 0.5, 1, 2, 4)}).T.rename_axis(f'lambda ({DONE[0]})')
display(sweep.round(2))

CATS = ['C0', 'C1', 'C2', 'C3', 'C4']


def class_means(gf):
    m = np.zeros((len(names), gf.shape[1]), np.float32)
    np.add.at(m, g_lab, gf)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


DCLS, DIAG = {}, {}
for n in DONE:
    gm = class_means(FEAT[n][1])
    DCLS[n] = 1 - gm @ gm.T, np.median(1 - (FEAT[n][0] * gm[q_lab]).sum(1))  # place-to-place d_F, tau
for v, top in RANK.items():
    d_cls, tau = DCLS[v.split('+')[0]]
    pred = g_lab[top[:, 0]]
    cat, err = error_types(pred, q_lab, xy, area, is_test)
    far = np.isin(cat, ['C2', 'C3', 'C4'])
    cause = np.where(~far, '', np.where(d_cls[q_lab, pred] < tau, 'ambiguous pair', 'domain gap'))
    DIAG[v] = pd.DataFrame({'query': [p for p, _ in qry.imgs], 'true': names[q_lab], 'pred': names[pred],
                            'cat': cat, 'err_m': err.round(1), 'cause': cause})
    DIAG[v].to_csv(f'{OUT}/diag_{v}.csv', index=False)

summary = pd.DataFrame({v: {**(d.cat.value_counts(normalize=True) * 100).reindex(CATS, fill_value=0).to_dict(),
                            'far errors: ambiguous %': 100 * (d.cause == 'ambiguous pair').sum() / max((d.cause != '').sum(), 1),
                            'median error m (wrong top-1)': d.err_m[d.cat != 'C0'].median()}
                        for v, d in DIAG.items()}).T.round(2)
summary.to_csv(f'{OUT}/error_types.csv')
display(summary)
for v in RANK:
    if v != DONE[0]:
        display(pd.crosstab(DIAG[DONE[0]].cat, DIAG[v].cat, rownames=[DONE[0]], colnames=[v]))
for n in DONE:
    qw, gw = DESC[n]
    err = np.flatnonzero((DIAG[n].cause != '').to_numpy())
    dw_pred = np.abs(qw[err] - gw[TOP[n][err, 0]]).sum(1)
    dw_true = np.array([np.abs(qw[i] - gw[g_lab == q_lab[i]]).sum(1).min() for i in err])
    rho = local_ratio(DCLS[n][0], xy)
    ok = (DIAG[n].cat == 'C0').to_numpy()
    print(f'{n}: d_W prefers the true place in {100 * (dw_true < dw_pred).mean():.1f}% of {len(err)} far errors; '
          f'median rho at the true place: correct {np.median(rho[q_lab[ok]]):.2e}, far errors {np.median(rho[q_lab[err]]):.2e} per m')

n = DONE[0]
d_cls, tau = DCLS[n]
d_g = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
iu = np.triu_indices(len(names), 1)
pg, pf = d_g[iu], d_cls[iu]
disc = np.median(pf[pg > 1000])  # typical d_F between places more than 1 km apart
print(f'{n}: tau = {tau:.3f}; ambiguous pairs (> 100 m apart, d_F < tau): {((pg > 100) & (pf < tau)).sum()}; '
      f'discontinuities (<= 30 m apart, d_F > {disc:.3f}): {((pg <= 30) & (pf > disc)).sum()} of {(pg <= 30).sum()} neighbour pairs')

fig, ax = plt.subplots(figsize=(7, 5))
*_, im = ax.hist2d(np.log10(pg + 1), pf, bins=120, norm=LogNorm())
ax.axvline(np.log10(31), ls='--', c='r')
ax.axhline(tau, ls='--', c='r')
ax.set(xlabel='log10(d_geo + 1), d_geo in m', ylabel='d_F = 1 - cos (place means)', title=f'{n}: all pairs of gallery places')
fig.colorbar(im, ax=ax)
fig.savefig(f'{OUT}/dgeo_dF_{n}.png', dpi=150)
plt.close(fig)

rho = local_ratio(d_cls, xy)
far_true = q_lab[(DIAG[n].cause != '').to_numpy()]
fig, axs = plt.subplots(1, len(test_areas), figsize=(5 * len(test_areas), 5), squeeze=False, layout='constrained')
for ax, a in zip(axs[0], test_areas):
    m = area == a
    sc = ax.scatter(*xy[m].T, c=rho[m], s=10, cmap='magma')
    e = far_true[area[far_true] == a]
    ax.scatter(*xy[e].T, marker='x', c='c', s=25, label='true place of a far error')
    ax.set_aspect('equal')
    ax.set_title(f'area {a}: local ratio rho (1/m)')
    ax.legend(loc='lower right', fontsize=8)
    fig.colorbar(sc, ax=ax, fraction=0.046)
fig.savefig(f'{OUT}/rho_map_{n}.png', dpi=150)
plt.close(fig)

first = {c: p for p, c in reversed(gal.imgs)}  # first gallery image of each place
cand = np.flatnonzero(pg > 100)
fig, axs = plt.subplots(2, 6, figsize=(18, 6.5))
for k, b in enumerate(cand[np.argsort(pf[cand])[:6]]):
    for row, c in enumerate((iu[0][b], iu[1][b])):
        axs[row, k].imshow(Image.open(first[c]))
        axs[row, k].axis('off')
    axs[0, k].set_title(f'{pg[b]:.0f} m apart, d_F = {pf[b]:.3f}', fontsize=9)
fig.suptitle(f'{n}: most similar pairs of places more than 100 m apart')
fig.savefig(f'{OUT}/ambiguous_pairs_{n}.png', dpi=120)
plt.close(fig)

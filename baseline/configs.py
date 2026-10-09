"""Train settings for every experiment. BASE mirrors the repo's train_test_local.sh; each run only overrides a few keys.
Paper numbers are from the DenseUAV paper (Tables V-VII)."""
import glob
import os

REPO_DIR = os.path.dirname(os.path.abspath(__file__))  # baseline/: DenseUAV code at commit b8de187 + Kaggle scripts
RESULTS_DIR = os.path.join(REPO_DIR, 'results')


def data_root():
    """DENSEUAV_ROOT, or the Kaggle input folder that holds Dense_GPS_ALL.txt."""
    if os.environ.get('DENSEUAV_ROOT'):
        return os.environ['DENSEUAV_ROOT']
    return os.path.dirname(glob.glob('/kaggle/input/**/Dense_GPS_ALL.txt', recursive=True)[0])


# train.py arguments, defaults of train_test_local.sh (num_worker = all CPUs)
BASE = dict(gpu_ids='0', num_worker=os.cpu_count(), lr=0.01, batchsize=16, sample_num=1, block=1,
            num_bottleneck=512, backbone='ViTS-224', head='SingleBranch', head_pool='avg',
            cls_loss='CELoss', feature_loss='WeightedSoftTripletLoss', kl_loss='KLLoss',
            h=224, w=224, load_from='no', ra='satellite', re='satellite', cj='no', rr='uav')

GLOBAL = dict(head_pool='global')  # class token, the paper's "Global" head
PAPER_T6 = dict(GLOBAL, feature_loss='HardMiningTripletLoss', kl_loss='no')  # Table VI as run in the paper


def run(name, paper, script='train.py', infer=False, **overrides):
    return dict(name=name, paper=paper, script=script, infer=infer, args={**BASE, **overrides})


EXPERIMENTS = {
    # Table VI, repo setup (CE + WeightedSoftTriplet + KL, lr 0.01)
    'heads_repo': [
        run('Global', {'R@1': 80.18, 'R@5': 93.99}, infer=True, **GLOBAL),
        run('FSRA_B2', {'R@1': 82.58, 'R@5': 94.94}, infer=True, **GLOBAL, head='FSRA', block=2),
        run('LPN_B2', {'R@1': 83.05, 'R@5': 94.89}, infer=True, **GLOBAL, head='LPN', block=2),
    ],
    # Table VI, paper setup (CE + HardMiningTriplet, no KL; lr 0.003 Global, 0.01 part-based)
    'heads_paper': [
        run('paper_Global', {'R@1': 80.18, 'R@5': 93.99}, infer=True, **PAPER_T6, lr=0.003),
        run('paper_FSRA_B2', {'R@1': 82.58, 'R@5': 94.94}, infer=True, **PAPER_T6, head='FSRA', block=2),
        run('paper_LPN_B2', {'R@1': 83.05, 'R@5': 94.89}, infer=True, **PAPER_T6, head='LPN', block=2),
    ],
    # Table V; ViT-S is heads_repo/Global
    'backbones': [
        run('DeiT-S', {'R@1': 71.77, 'R@5': 89.70}, infer=True, **GLOBAL, backbone='DeitS-224'),
        run('Swinv2-T', {'R@1': 77.99, 'R@5': 92.49}, infer=True, **GLOBAL, backbone='Swinv2T-256',
            head='SingleBranchSwin', h=256, w=256),
    ],
    # Table VII, repo head (SingleBranch avg pool)
    'losses': [
        run('loss_ce', {'R@1': 76.28, 'SDM@1': 81.23, 'SDM@3': 79.25, 'SDM@5': 74.39, 'SDM@10': 63.0},
            feature_loss='no', kl_loss='no'),
        run('loss_focal', {'R@1': 73.19, 'SDM@1': 79.14, 'SDM@3': 77.04, 'SDM@5': 72.46, 'SDM@10': 61.98},
            cls_loss='FocalLoss', feature_loss='no', kl_loss='no'),
        run('loss_ce_triplet', {'R@1': 79.19, 'SDM@1': 83.63, 'SDM@3': 81.2, 'SDM@5': 76.9, 'SDM@10': 65.26},
            feature_loss='TripletLoss', kl_loss='no'),
        run('loss_ce_hardtriplet', {'R@1': 80.18, 'SDM@1': 84.39, 'SDM@3': 82.51, 'SDM@5': 78.02, 'SDM@10': 66.46},
            feature_loss='HardMiningTripletLoss', kl_loss='no'),
        run('loss_ce_samedomain', {'R@1': 81.17, 'SDM@1': 85.09, 'SDM@3': 82.93, 'SDM@5': 78.51, 'SDM@10': 67.0},
            feature_loss='SameDomainTripletLoss', kl_loss='no'),
        run('loss_ce_softtriplet', {'R@1': 81.17, 'SDM@1': 85.22, 'SDM@3': 83.1, 'SDM@5': 78.89, 'SDM@10': 67.26},
            kl_loss='no'),
        run('loss_ce_softtriplet_kl', {'R@1': 83.01, 'SDM@1': 86.5, 'SDM@3': 84.5, 'SDM@5': 80.44, 'SDM@10': 68.49}),
    ],
    # CT2: baseline vs neighbour-aware sampler + local consistency loss (train_geo.py)
    'ct': [
        run('global', {'R@1': 80.18, 'SDM@1': 84.39}, **GLOBAL),
        run('geo_lc', {}, script='train_geo.py', **GLOBAL),
    ],
}

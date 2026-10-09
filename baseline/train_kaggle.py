"""Train and test the runs of one experiment, the same steps as the repo's train_test_local.sh:
train -> test.py -> evaluate_gpu.py (Recall/AP) -> evaluateDistance.py (SDM@K) [-> tool/get_inference_time.py].
Usage: python train_kaggle.py <experiment> [run names...]   e.g. python train_kaggle.py heads_repo LPN_B2"""
import os
import subprocess
import sys

from configs import EXPERIMENTS, REPO_DIR, data_root


def sh(cmd, cwd, out=None, env=None):
    print('+', ' '.join(map(str, cmd)), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=cwd, check=True, stdout=out, env=env)


def train_test(r, root):
    name, a = r['name'], r['args']
    ckpt = f'{REPO_DIR}/checkpoints/{name}'
    py = sys.executable
    if os.path.exists(f'{ckpt}/results.txt'):
        print(name, 'already done, skipping')
    else:
        sh([py, r['script'], '--name', name, '--data_dir', f'{root}/train',
            *[x for k, v in a.items() for x in (f'--{k}', v)]], REPO_DIR)
        sh([py, 'test.py', '--name', name, '--test_dir', f'{root}/test', '--gpu_ids', a['gpu_ids'],
            '--num_worker', a['num_worker']], ckpt)
        sh([py, 'evaluate_gpu.py'], ckpt)
        sh([py, 'evaluateDistance.py', '--root_dir', root], ckpt)
    if r['infer'] and not os.path.exists(f'{ckpt}/infer.txt'):
        with open(f'{ckpt}/infer.txt', 'w') as f:
            sh([py, 'get_inference_time.py', '--test_h', a['h'], '--test_w', a['w']], f'{ckpt}/tool', f,
               {**os.environ, 'PYTHONPATH': '..'})


if __name__ == '__main__':
    exp, only = sys.argv[1], sys.argv[2:]
    root = data_root()
    for r in EXPERIMENTS[exp]:
        if not only or r['name'] in only:
            train_test(r, root)

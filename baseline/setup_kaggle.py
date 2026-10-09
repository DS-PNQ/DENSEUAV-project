"""Install the baseline's requirements on Kaggle, check the dataset and self-test the CT helpers.
The code in baseline/ is DenseUAV at commit b8de187 with the compatibility fixes already applied.
Usage: python setup_kaggle.py"""
import os
import subprocess
import sys

from configs import REPO_DIR, data_root

subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', f'{REPO_DIR}/requirments.txt'], check=True)
root = data_root()
# class counts from the repo README
for d, n in [('train/drone', 2256), ('train/satellite', 2256), ('test/query_drone', 777), ('test/gallery_satellite', 3033)]:
    assert len(os.listdir(f'{root}/{d}')) == n, d
print('data:', root)
os.makedirs(f'{REPO_DIR}/checkpoints', exist_ok=True)
subprocess.run([sys.executable, 'geo_utils.py'], cwd=REPO_DIR, check=True)

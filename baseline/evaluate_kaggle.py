"""Collect the repo's metrics of one experiment into a table next to the paper's numbers.
Usage: python evaluate_kaggle.py <experiment>   (writes results/<experiment>.csv)"""
import json
import os
import re
import sys

import pandas as pd

from configs import EXPERIMENTS, REPO_DIR, RESULTS_DIR


def metrics(name):
    """results.txt (evaluate_gpu.py), SDM@K(1,100).json (evaluateDistance.py) and infer.txt if present."""
    B = f'{REPO_DIR}/checkpoints/{name}'
    txt = open(f'{B}/results.txt').read()
    if os.path.exists(f'{B}/infer.txt'):
        txt += open(f'{B}/infer.txt').read()
    r = dict(re.findall(r'([\w@]+)[:=]([\d.]+[MGs]?)', txt))
    sdm = json.load(open(f'{B}/SDM@K(1,100).json'))
    row = {k: r[k] for k in ('Params', 'Macs', 'InferTime') if k in r}
    row.update({f'R@{k}': float(r[f'Recall@{k}']) for k in (1, 5, 10)}, AP=float(r['AP']))
    row.update({f'SDM@{k}': round(sdm[str(k)] * 100, 2) for k in (1, 3, 5, 10)})
    return row


if __name__ == '__main__':
    exp = sys.argv[1]
    rows = {}
    for r in EXPERIMENTS[exp]:
        if not os.path.exists(f"{REPO_DIR}/checkpoints/{r['name']}/results.txt"):
            print(r['name'], 'not run yet')
            continue
        rows[r['name']] = {**metrics(r['name']), **{f'paper {k}': v for k, v in r['paper'].items()}}
    df = pd.DataFrame(rows).T
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(f'{RESULTS_DIR}/{exp}.csv')
    print(df.to_string())

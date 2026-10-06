import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('route,estimator,topk', [
    ('sequence', 'binary', 2), ('sequence', 'full', 2),
    ('sequence', 'topk', 2), ('sequence', 'topk', 128),
    ('token', 'binary', 2),
    ('token', 'topk', 2), ('token', 'topk', 128),
    ('token', 'full', 2),
    ('sequence', 'mc', 2), ('token', 'mc', 2),
])
def test_real_autoregressive_updates_and_stale_reuse(route, estimator, topk):
    command = [sys.executable, str(ROOT / 'examples/train_toy.py'), '--updates', '4', '--top-k', str(topk),
               '--mc-samples', '2' if route == 'sequence' else '1']
    if (route, estimator) != ('token', 'mc'):
        command += ['--route', route, '--kl-estimator', estimator]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(rows) == 4
    assert all(r['route'] == route and r['kl_estimator'] == estimator for r in rows)
    assert [r['sampler_version'] for r in rows] == [0] * 4
    assert rows[-1]['lag'] == 3
    if estimator == 'mc':
        assert all(r['mc_samples'] == (2 if route == 'sequence' else 1) for r in rows)
        assert all(r['head_size'] is None for r in rows)
    assert all(r['update_norm'] > 0 for r in rows)
    assert rows[0]['surrogate'] != rows[1]['surrogate']

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from klpo import klpo_sequence_loss, klpo_token_loss
from klpo.molt import KLPOLoss

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('check_molt', ROOT / 'scripts/check_molt.py')
BACKEND = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BACKEND)


@pytest.mark.parametrize('route', ['sequence', 'token'])
def test_microbatch_and_uneven_dp_shard_gradients_match_global_batch(route):
    current = torch.tensor([[-1., -2., -.7], [-.5, -.2, -1.], [-.4, -2., -1.], [-1.3, -2., -3.]],
                           dtype=torch.float64, requires_grad=True)
    old = current.detach() - .3
    rewards = torch.tensor([1., -.2, 0., .4], dtype=torch.float64)
    mask = torch.tensor([[True, False, False], [True, True, False], [True, True, True], [True, False, True]])
    if route == 'sequence':
        baseline, _ = klpo_sequence_loss(current, old, rewards, mask, beta=.2)
    else:
        baseline, _ = klpo_token_loss(current, old, rewards, mask, kl_estimator='binary', beta=.2)
    expected, = torch.autograd.grad(baseline, current)
    adapter = KLPOLoss(beta=.2, route=route, kl_estimator="binary")
    # Two DP ranks, unequal local sizes, one rank split into two microbatches.
    losses = []
    for part in (slice(0, 1), slice(1, 2), slice(2, 4)):
        result = adapter(current[part], torch.full_like(old[part], -500.), torch.full_like(old[part], 99.),
            action_mask=mask[part], rollout_log_probs=old[part], dp_size=2,
            global_batch_size=4, rewards=rewards[part])
        assert len(result) == 6
        assert all(not x.requires_grad for x in result[1:])
        losses.append(result[0])
    # DDP averages gradients over its two data-parallel ranks.
    (sum(losses) / 2).backward()
    torch.testing.assert_close(current.grad, expected)


@pytest.mark.parametrize('failure', ['no_batch', 'no_sampler', 'bad_batch', 'bad_dp'])
def test_adapter_rejects_missing_contract(failure):
    kwargs = dict(action_mask=torch.ones(1, 2, dtype=torch.bool), rollout_log_probs=torch.full((1, 2), -2.),
                  global_batch_size=1, dp_size=1, rewards=torch.ones(1))
    if failure == 'no_batch': kwargs['global_batch_size'] = None
    if failure == 'no_sampler': kwargs['rollout_log_probs'] = None
    if failure == 'bad_batch': kwargs['global_batch_size'] = 0
    if failure == 'bad_dp': kwargs['dp_size'] = 0
    with pytest.raises(ValueError):
        KLPOLoss()(torch.full((1, 2), -1.), None, None, **kwargs)


@pytest.mark.parametrize('recipe', ['r1', 'qwen_math'])
@pytest.mark.parametrize('route', ['sequence', 'token'])
def test_training_dry_run_uses_actual_klpo_path(recipe, route):
    command = [sys.executable, str(ROOT / 'scripts/train_molt.py'),
        '--recipe', recipe, '--molt-path', '/tmp/molt path', '--model', 'model',
        '--train-data', '/tmp/train', '--eval-data', '/tmp/eval', '--beta', '.3', '--dry-run']
    if route != 'token':
        command += ['--route', route]
    result = subprocess.run(command, text=True, capture_output=True, check=True)
    for flag in ('--actor.loss_mode klpo', '--actor.klpo_beta 0.3', '--actor.klpo_kl_estimator mc', '--actor.klpo_mc_samples 128', '--train.async_queue_size 4',
                 '--algo.advantage.is_correction_level off', '--algo.advantage.no_whiten',
                 '--algo.kl.init_coef 0', '--rollout.n_samples_per_prompt 1', '--train.max_epochs 1',
                 f'--actor.klpo_route {route}'):
        assert flag in result.stdout
    assert 'is_correction_threshold' not in result.stdout and 'flash_reinforce' not in result.stdout
    assert '--train.force_sync_mode' not in result.stdout
    assert '--train.force_on_policy' not in result.stdout


def test_training_sync_is_an_explicit_fallback():
    command = [sys.executable, str(ROOT / 'scripts/train_molt.py'),
        '--molt-path', '/tmp/molt', '--model', 'model', '--train-data', '/tmp/train',
        '--eval-data', '/tmp/eval', '--sync', '--async-queue-size', '8', '--dry-run']
    result = subprocess.run(command, text=True, capture_output=True, check=True)
    assert '--train.async_queue_size 1' in result.stdout
    assert '--train.force_sync_mode' in result.stdout
    assert '--train.force_on_policy' in result.stdout


def test_native_backend_version_and_worker_contract():
    root = os.environ.get('MOLT_SOURCE_PATH')
    if root is None:
        pytest.skip('Set MOLT_SOURCE_PATH to the native labs-molt fork')
    root = Path(root)
    assert 'KLPO_API_VERSION = 2' in (root / 'molt/__init__.py').read_text()
    actor = (root / 'molt/trainer/workers/policy_actor.py').read_text()
    for contract in ('from klpo.molt import KLPOLoss', 'kl_estimator=self.args.actor.klpo_kl_estimator',
                     '"rewards": experience.rewards', 'global_batch_size=batch_num_seqs',
                     'scale_loss_by_accumulation=False', 'experience.kl_token_ids', 'experience.kl_log_probs'):
        assert contract in actor
    assert not (ROOT / 'scripts/prepare_molt.py').exists()


@pytest.mark.parametrize('route', ['sequence', 'token'])
@pytest.mark.parametrize('estimator', ['binary', 'topk', 'mc', 'full'])
def test_native_adapter_all_routes_match_core_gradients(route, estimator):
    import klpo

    torch.manual_seed(3)
    logits = torch.randn(4, 3, 5, dtype=torch.float64, requires_grad=True)
    p = logits.log_softmax(-1)
    q = torch.randn_like(logits).log_softmax(-1)
    actions = torch.tensor([[0, 1, 2], [2, 3, 4], [4, 0, 1], [3, 2, 0]])
    logp = p.gather(-1, actions[..., None]).squeeze(-1)
    logq = q.gather(-1, actions[..., None]).squeeze(-1)
    mask = torch.tensor([[1, 1, 0], [1, 0, 0], [1, 1, 1], [1, 0, 1]], dtype=torch.bool)
    rewards = torch.tensor([1., -.2, .8, 0.], dtype=logits.dtype)
    paux = qaux = None
    options = {'beta': .2}
    if estimator == 'topk':
        ids = q.topk(2, -1).indices
        paux, qaux = p.gather(-1, ids), q.gather(-1, ids)
        options.update(conditional_log_probs=paux, behavior_conditional_log_probs=qaux)
    elif estimator == 'mc':
        ids = torch.tensor([1, 1, 3]).expand(4, 3, 3)
        paux, qaux = p.gather(-1, ids), q.gather(-1, ids)
        options.update(mc_log_probs=paux, behavior_mc_log_probs=qaux)
    elif estimator == 'full':
        paux, qaux = p, q
        if route == 'sequence':
            options.update(full_log_probs=p, behavior_full_log_probs=q)
        else:
            options.update(conditional_log_probs=p, behavior_conditional_log_probs=q)
    if route == 'sequence':
        fn = {'binary': klpo.klpo_sequence_loss, 'topk': klpo.klpo_sequence_topk_loss,
              'mc': klpo.klpo_sequence_mc_loss, 'full': klpo.klpo_sequence_full_loss}[estimator]
    else:
        fn = klpo.klpo_token_loss
        options['kl_estimator'] = estimator
    baseline, _ = fn(logp, logq, rewards, mask, **options)
    expected, = torch.autograd.grad(baseline, logits, retain_graph=True)
    adapter = (KLPOLoss(beta=.2) if (route, estimator) == ('token', 'mc')
               else KLPOLoss(beta=.2, route=route, kl_estimator=estimator))
    losses = []
    for part in (slice(0, 1), slice(1, 2), slice(2, 4)):
        result = adapter(logp[part], None, None, rewards=rewards[part], action_mask=mask[part],
                         rollout_log_probs=logq[part], dp_size=2, global_batch_size=4,
                         kl_log_probs=paux[part] if paux is not None else None,
                         behavior_kl_log_probs=qaux[part] if qaux is not None else None,
                         full_vocabulary=estimator == 'full')
        losses.append(result[0])
    got, = torch.autograd.grad(sum(losses) / 2, logits)
    torch.testing.assert_close(got, expected)


@pytest.mark.parametrize('route', ['sequence', 'token'])
@pytest.mark.parametrize('estimator', ['binary', 'topk', 'mc', 'full'])
def test_launcher_selects_estimator_without_source_patches(route, estimator):
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/train_molt.py'),
        '--molt-path', '/tmp/native-molt', '--model', 'model', '--train-data', '/tmp/train',
        '--eval-data', '/tmp/eval', '--route', route, '--kl-estimator', estimator,
        '--top-k', '64', '--mc-samples', '2', '--dry-run'], text=True, capture_output=True, check=True)
    assert f'--actor.klpo_kl_estimator {estimator}' in result.stdout
    assert '--actor.klpo_top_k 64' in result.stdout
    assert '--actor.klpo_mc_samples 2' in result.stdout

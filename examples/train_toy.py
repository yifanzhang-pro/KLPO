"""CPU autoregressive KLPO training with stale samplers, replay, and no downloads.

The policy generates digits 1..3 or EOS=0. A terminal verifier rewards a sum
modulo three matching the prompt. The state includes prompt, position and the
running sum; it is deterministic. Max-length termination is part of this toy
environment (not an unfinished real-world rollout).
"""

import argparse
import json
import math

import torch

from klpo import (kl_budget_scale, klpo_sequence_full_loss, klpo_sequence_loss,
                  klpo_sequence_mc_loss, klpo_token_loss, klpo_sequence_topk_loss, predicted_kl)


@torch.no_grad()
def collect(sampler, batch_size, max_tokens, sampler_version):
    states = torch.zeros(batch_size, max_tokens, dtype=torch.long)
    actions = torch.zeros_like(states)
    mask = torch.zeros_like(states, dtype=torch.bool)
    full = torch.zeros(batch_size, max_tokens, 4, dtype=sampler.dtype)
    rewards = torch.zeros(batch_size, dtype=sampler.dtype)
    for i in range(batch_size):
        prompt, checksum = i % 3, 0
        for t in range(max_tokens):
            state = (prompt * max_tokens + t) * 3 + checksum
            lp = sampler[state].log_softmax(-1)
            action = torch.multinomial(lp.exp(), 1).item()
            states[i, t], actions[i, t], mask[i, t] = state, action, True
            full[i, t] = lp
            if action == 0:
                break
            checksum = (checksum + action) % 3
        rewards[i] = float(checksum == prompt)
    return dict(states=states, actions=actions, mask=mask, full=full, rewards=rewards,
                sampler_version=sampler_version)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--route', choices=['sequence', 'token'], default='token')
    parser.add_argument('--kl-estimator', choices=['binary', 'full', 'topk', 'mc'], default='mc',
                        help='Default: mc (Monte Carlo KL) for both routes')
    parser.add_argument('--updates', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=12)
    parser.add_argument('--max-tokens', type=int, default=4)
    parser.add_argument('--reuse', type=int, default=2, help='Learner steps per collected batch')
    parser.add_argument('--publish-every', type=int, default=4, help='Learner updates between sampler snapshots')
    parser.add_argument('--top-k', type=int, default=128, help='Capped to the toy vocabulary size of four')
    parser.add_argument('--mc-samples', type=int, default=128,
                        help='IID MC-KL draws per prefix: M>=2 for sequence, M>=1 for token; not capped to vocabulary')
    parser.add_argument('--beta', type=float, default=.1)
    parser.add_argument('--lr', type=float, default=.03)
    parser.add_argument('--kl-budget', type=float, default=None)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    if min(args.updates, args.batch_size, args.max_tokens, args.reuse, args.publish_every, args.top_k, args.mc_samples) < 1:
        parser.error('counts must be positive')
    if not math.isfinite(args.lr) or args.lr <= 0 or not math.isfinite(args.beta) or args.beta <= 0:
        parser.error('lr and beta must be positive and finite')
    if args.kl_budget is not None and (not math.isfinite(args.kl_budget) or args.kl_budget < 0):
        parser.error('kl-budget must be nonnegative and finite')
    estimator = args.kl_estimator
    if estimator == 'mc' and args.route == 'sequence' and args.mc_samples < 2:
        parser.error('sequence MC-KL needs --mc-samples >= 2')
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    logits = torch.nn.Parameter(torch.randn(3 * args.max_tokens * 3, 4, dtype=torch.float64) * .1)
    optimizer = torch.optim.AdamW([logits], lr=args.lr, weight_decay=.1)
    sampler, version, batch = logits.detach().clone(), 0, None
    for step in range(args.updates):
        if step % args.publish_every == 0:
            sampler, version = logits.detach().clone(), step
        if step % args.reuse == 0:
            batch = collect(sampler, args.batch_size, args.max_tokens, version)
            if estimator in {'topk', 'full'}:
                # The rollout action can be outside the TopK-KL head.
                k = 4 if estimator == 'full' else min(args.top_k, 4)
                head_logps, head_ids = batch['full'].topk(k, dim=-1)
            old_action = batch['full'].gather(-1, batch['actions'][..., None]).squeeze(-1)
        full = logits[batch['states']].log_softmax(-1)
        current = full.gather(-1, batch['actions'][..., None]).squeeze(-1)
        inputs = (current, old_action, batch['rewards'], batch['mask'])
        if estimator == 'mc':
            # Fresh auxiliary tokens, independent of the COMPLETE rollout and
            # previous learner updates. Resample from the historical conditionals,
            # never the current trainer or a newly published sampler checkpoint.
            mc_ids = torch.multinomial(batch['full'].exp().reshape(-1, 4),
                                       args.mc_samples, replacement=True)
            mc_ids = mc_ids.reshape(*batch['mask'].shape, args.mc_samples)
            records = dict(mc_log_probs=full.gather(-1, mc_ids),
                           behavior_mc_log_probs=batch['full'].gather(-1, mc_ids))
            if args.route == 'sequence':
                loss, stats = klpo_sequence_mc_loss(*inputs, **records, beta=args.beta)
            else:
                loss, stats = klpo_token_loss(*inputs, **records, kl_estimator='mc', beta=args.beta)
        elif args.route == 'sequence' and estimator == 'binary':
            loss, stats = klpo_sequence_loss(*inputs, beta=args.beta)
        elif args.route == 'sequence' and estimator == 'full':
            loss, stats = klpo_sequence_full_loss(*inputs, full_log_probs=full,
                behavior_full_log_probs=batch['full'], beta=args.beta)
        elif args.route == 'sequence':
            loss, stats = klpo_sequence_topk_loss(*inputs, conditional_log_probs=full.gather(-1, head_ids),
                behavior_conditional_log_probs=head_logps, full_vocabulary=k == 4, beta=args.beta)
        elif estimator == 'binary':
            loss, stats = klpo_token_loss(*inputs, kl_estimator='binary', beta=args.beta)
        else:
            loss, stats = klpo_token_loss(*inputs, conditional_log_probs=full.gather(-1, head_ids),
                behavior_conditional_log_probs=head_logps, kl_estimator=estimator, full_vocabulary=k == 4, beta=args.beta)
        optimizer.zero_grad()
        loss.backward()
        before = logits.detach().clone()
        optimizer.step()
        # Complete AdamW proposal, including LR, moments and weight decay.
        displacement = logits.detach() - before
        alpha, prediction, realized = 1., None, None
        if args.kl_budget is not None:
            with torch.no_grad():
                logits.copy_(before)
            probe = lambda parameters: parameters[batch['states']]
            z, dz = torch.func.jvp(probe, (before,), (displacement,))
            prediction = predicted_kl(z, dz, batch['mask'])
            alpha = kl_budget_scale(prediction, args.kl_budget)
            with torch.no_grad():
                logits.copy_(before + alpha * displacement)
                old_log = z[batch['mask']].log_softmax(-1)
                new_log = probe(logits)[batch['mask']].log_softmax(-1)
                realized = (old_log.exp() * (old_log - new_log)).sum(-1).mean().item()
            prediction = prediction.item()
        row = dict(route=args.route, kl_estimator=estimator, step=step + 1, sampler_version=batch['sampler_version'],
                   mc_samples=args.mc_samples if estimator == 'mc' else None,
                   head_size=k if estimator == 'topk' else None,
                   lag=step - batch['sampler_version'], reward=batch['rewards'].mean().item(),
                   surrogate=loss.item(), update_norm=(logits.detach() - before).norm().item(),
                   scale=alpha, predicted_kl=prediction, realized_kl=realized,
                   probe_failed=prediction is not None and (not math.isfinite(prediction) or prediction <= 0))
        if 'regression_loss' in stats:
            row['regression_loss'] = stats['regression_loss'].item()
        print(json.dumps(row))


if __name__ == '__main__':
    main()

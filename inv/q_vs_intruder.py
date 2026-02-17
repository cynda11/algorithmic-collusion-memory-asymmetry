#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Q-learning (1-step memory) vs simple intruders (from t=0), Klein-style sequential pricing.

Players
-------
- Firm 1: tabular Q-learning with 1-step memory (state = opponent's last price).
- Firm 2: an "intruder" algorithm (chosen by --intruder) that maximizes its *own* profit.

Intruders implemented
---------------------
1) mbr   : myopic best response (chooses the price that maximizes current-period profit).
2) exp3  : stateless adversarial bandit EXP3 over the discrete price grid (arms = prices).
3) rbf   : rule-based follower (simple collusive-style follower): matches Firm 1's current price.
          (It is intentionally different from MBR and EXP3 to provide behavioural diversity.)

Environment
-----------
- Discrete price grid {0, 1/k, ..., 1}
- Alternating moves (Firm 1 moves at even t, Firm 2 moves at odd t)
- Demand and profit as in the baseline runner:
    if p_i<p_j: D_i=1-p_i; if p_i=p_j: D_i=0.5(1-p_i); else 0.
    profit_i = p_i * D_i.

Learning (Firm 1)
-----------------
Klein-style two-step target with correct timing:
  Q(s,a) <- (1-α)Q(s,a) + α[ π_t + δ·π_{t+1} + δ^2·max_{a'} Q(s',a') ]
where π_{t+1} uses Firm 1's own price chosen at time t.

Quick run example:
  python q_vs_intruder.py --intruder mbr  --T 20000 --n_seeds 50 --csv out_mbr.csv
  python q_vs_intruder.py --intruder exp3 --T 20000 --n_seeds 50 --csv out_exp3.csv
  python q_vs_intruder.py --intruder rbf  --T 20000 --n_seeds 50 --csv out_rbf.csv

Output CSV includes per-firm late-phase averages:
  avg_price_f1, avg_price_f2, avg_profit_f1, avg_profit_f2
"""

import argparse
import math
import numpy as np
import pandas as pd


def demand(p_i: float, p_j: float) -> float:
    if p_i < p_j:
        return 1.0 - p_i
    if p_i == p_j:
        return 0.5 * (1.0 - p_i)
    return 0.0


def profit_from_indices(prices: np.ndarray, i_idx: int, j_idx: int) -> float:
    p_i = float(prices[i_idx])
    p_j = float(prices[j_idx])
    return p_i * demand(p_i, p_j)


def pick_mbr(prices: np.ndarray, opp_idx: int) -> int:
    """Myopic best response: argmax_a profit(a, opp)."""
    n = len(prices)
    vals = np.array([profit_from_indices(prices, a, opp_idx) for a in range(n)])
    best = np.flatnonzero(vals == vals.max())
    return int(np.random.choice(best))


def exp3_init(n: int):
    return np.ones(n, dtype=float)


def exp3_probs(weights: np.ndarray, gamma: float):
    n = len(weights)
    w = weights / weights.sum()
    return (1.0 - gamma) * w + gamma / n


def exp3_sample_action(rng: np.random.Generator, weights: np.ndarray, gamma: float) -> tuple[int, np.ndarray]:
    probs = exp3_probs(weights, gamma)
    a = int(rng.choice(len(weights), p=probs))
    return a, probs


def exp3_update(weights, a, reward, probs, gamma, reward_scale):
    n = len(weights)
    rhat = max(0.0, min(1.0, reward / reward_scale))
    xhat = rhat / max(1e-12, probs[a])

    # --- stabilize exponent ---
    exp_arg = (gamma * xhat) / n
    exp_arg = min(exp_arg, 40)    # prevents overflow

    weights[a] *= math.exp(exp_arg)

    # --- renormalize weights ---
    s = weights.sum()
    if not np.isfinite(s) or s == 0:
        weights[:] = 1.0
    else:
        weights /= s


def self_play_one(seed: int, T: int, k: int, alpha: float, delta: float,
                  intruder: str, exp3_gamma: float, tie_break: str = 'random') -> dict:

    rng = np.random.default_rng(seed)

    prices = np.linspace(0.0, 1.0, k + 1)
    n = len(prices)

    # Firm 1 exploration schedule (same structure as baseline)
    theta = 1.0 - (1e-6) ** (1.0 / T)

    # Firm 1 Q-table: state = opponent last price index, action = own price index
    Q1 = np.zeros((n, n))

    # Intruder internal state (only for EXP3)
    exp3_w = exp3_init(n) if intruder == 'exp3' else None

    # Initial actions (price indices)
    p1 = int(rng.integers(n))
    p2 = int(rng.integers(n))

    # Pending update for Firm 1: (state_idx, action_idx, immediate_profit)
    pend1 = None

    hist_p1 = np.empty(T)
    hist_p2 = np.empty(T)
    hist_pi1 = np.empty(T)
    hist_pi2 = np.empty(T)

    def choose(Qrow: np.ndarray) -> int:
        if tie_break == 'first':
            return int(np.argmax(Qrow))
        m = Qrow.max()
        best = np.flatnonzero(Qrow == m)
        return int(rng.choice(best))

    # reward scale for EXP3: max possible single-period profit is 0.25 (undercut and take whole market at p=0.5)
    reward_scale = 0.25

    for t in range(T):
        if t % 2 == 0:
            # ---------------- Firm 1 moves ----------------
            eps = (1.0 - theta) ** t
            s1 = p2
            a1 = int(rng.integers(n)) if rng.random() < eps else choose(Q1[s1])
            imm1 = profit_from_indices(prices, a1, p2)

            pend1 = (s1, a1, imm1)
            p1 = a1

            pi1_now = imm1
            pi2_now = profit_from_indices(prices, p2, p1)

        else:
            # ---------------- Intruder (Firm 2) moves ----------------
            if intruder == 'mbr':
                a2 = pick_mbr(prices, p1)
                probs = None
            elif intruder == 'rbf':
                # simple follower: match Firm 1's current price (behaviourally distinct from MBR/EXP3)
                a2 = int(p1)
                probs = None
            elif intruder == 'exp3':
                a2, probs = exp3_sample_action(rng, exp3_w, exp3_gamma)
            else:
                raise ValueError(f'Unknown intruder: {intruder}')

            imm2 = profit_from_indices(prices, a2, p1)

            # Update Firm 1 using its pending from previous move
            if pend1 is not None:
                s1, a1_prev, imm1_prev = pend1

                # next-period profit for Firm 1 uses its own price a1_prev vs new Firm2 price a2
                next_profit1 = profit_from_indices(prices, a1_prev, a2)

                # next state for Firm 1 after Firm 2 moves is opponent_last=a2
                best_future = Q1[a2].max()

                target = imm1_prev + delta * next_profit1 + (delta ** 2) * best_future
                Q1[s1, a1_prev] = (1.0 - alpha) * Q1[s1, a1_prev] + alpha * target

                pend1 = None

            # EXP3 update (uses realized reward)
            if intruder == 'exp3':
                exp3_update(exp3_w, a2, imm2, probs, exp3_gamma, reward_scale)

            p2 = a2

            pi1_now = profit_from_indices(prices, p1, p2)
            pi2_now = imm2

        hist_p1[t] = prices[p1]
        hist_p2[t] = prices[p2]
        hist_pi1[t] = pi1_now
        hist_pi2[t] = pi2_now

    # ---------------- Late-phase summary (last 10%) ----------------
    tail = int(0.1 * T)
    p1_last = hist_p1[-tail:]
    p2_last = hist_p2[-tail:]
    t_last = np.arange(T - tail, T)

    market_last = 0.5 * (p1_last + p2_last)

    mono_price = float(prices[int(np.argmin(np.abs(prices - 0.5)))])
    share_state_05_05 = float(np.mean((p1_last == mono_price) & (p2_last == mono_price)))

    mask_f1 = (t_last % 2 == 0)
    idx_even = np.where(mask_f1)[0]
    idx_even = idx_even[idx_even + 1 < len(t_last)]
    share_joint_consecutive_05 = (
        float(np.mean((p1_last[idx_even] == mono_price) & (p2_last[idx_even + 1] == mono_price)))
        if len(idx_even) else 0.0
    )

    pair_series = pd.Series(list(zip(np.round(p1_last, 6), np.round(p2_last, 6))))
    mode_pair = pair_series.value_counts().idxmax()
    share_mode_pair = float(pair_series.value_counts(normalize=True).max())

    tick = 1.0 / k
    diffs = np.diff(market_last)
    jump_count_large = int(np.sum(np.abs(diffs) > tick + 1e-12)) if len(diffs) else 0

    std_market = float(np.std(market_last))
    cycle_amp = float(np.max(market_last) - np.min(market_last))
    share_equal = float(np.mean(p1_last == p2_last))

    if share_state_05_05 > 0.8 and std_market < 0.02:
        regime = 'monopoly_regime'
    elif share_equal > 0.8 and std_market < 0.02:
        regime = 'focal_equal_price'
    elif cycle_amp > 0.15:
        regime = 'cycling/asymmetric'
    else:
        regime = 'mixed'

    avg_price = float(np.mean(market_last))
    avg_price_f1 = float(np.mean(p1_last))
    avg_price_f2 = float(np.mean(p2_last))
    avg_profit_f1 = float(np.mean(hist_pi1[-tail:]))
    avg_profit_f2 = float(np.mean(hist_pi2[-tail:]))
    avg_profit = float(0.5 * (avg_profit_f1 + avg_profit_f2))

    return {
        'seed': seed,
        'T': T,
        'k': k,
        'intruder': intruder,
        'avg_price': avg_price,
        'avg_price_f1': avg_price_f1,
        'avg_price_f2': avg_price_f2,
        'avg_profit': avg_profit,
        'avg_profit_f1': avg_profit_f1,
        'avg_profit_f2': avg_profit_f2,
        'share_state_0.5_0.5': share_state_05_05,
        'share_joint_consecutive_0.5': share_joint_consecutive_05,
        'mode_pair': str(mode_pair),
        'share_mode_pair': share_mode_pair,
        'std_market': std_market,
        'cycle_amp': cycle_amp,
        'jump_count_large': jump_count_large,
        'regime': regime,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--intruder', type=str, required=True, choices=['mbr','exp3','rbf'])
    ap.add_argument('--T', type=int, default=50_000)
    ap.add_argument('--k', type=int, default=6)
    ap.add_argument('--alpha', type=float, default=0.3)
    ap.add_argument('--delta', type=float, default=0.95)
    ap.add_argument('--n_seeds', type=int, default=50)
    ap.add_argument('--seed_start', type=int, default=1)
    ap.add_argument('--exp3_gamma', type=float, default=0.07)
    ap.add_argument('--csv', type=str, default='results_q_vs_intruder.csv')
    args = ap.parse_args()

    seeds = range(args.seed_start, args.seed_start + args.n_seeds)
    rows = []

    for i, s in enumerate(seeds, start=1):
        rows.append(self_play_one(s, args.T, args.k, args.alpha, args.delta,
                                 intruder=args.intruder, exp3_gamma=args.exp3_gamma))
        if i % max(1, args.n_seeds // 10) == 0:
            print(f'progress {i}/{args.n_seeds}')

    df = pd.DataFrame(rows)
    df.to_csv(args.csv, index=False)

    print('\n=== AGGREGATE SUMMARY (late phase) ===')
    print(f"runs: {len(df)} | intruder={args.intruder} | T={args.T} | k={args.k} | alpha={args.alpha} | delta={args.delta}")

    print(f"avg(avg_price): {df['avg_price'].mean():.4f} | median: {df['avg_price'].median():.4f}")
    print(f"avg(avg_profit): {df['avg_profit'].mean():.4f} | median: {df['avg_profit'].median():.4f}")

    print(f"avg(avg_profit_f1): {df['avg_profit_f1'].mean():.4f} | median: {df['avg_profit_f1'].median():.4f}")
    print(f"avg(avg_profit_f2): {df['avg_profit_f2'].mean():.4f} | median: {df['avg_profit_f2'].median():.4f}")
    print(f"avg(avg_price_f1):  {df['avg_price_f1'].mean():.4f} | median: {df['avg_price_f1'].median():.4f}")
    print(f"avg(avg_price_f2):  {df['avg_price_f2'].mean():.4f} | median: {df['avg_price_f2'].median():.4f}")

    print(f"share intruder_profit>q_profit: {(df['avg_profit_f2'] > df['avg_profit_f1']).mean():.2%}")
    print(f"share monopoly_regime: {(df['regime']=='monopoly_regime').mean():.2%}")
    print(f"mean share_state_0.5_0.5: {df['share_state_0.5_0.5'].mean():.2%}")

    top = df.sort_values('share_state_0.5_0.5', ascending=False).head(10)
    print('\nTop 10 by share_state_0.5_0.5:')
    cols = ['seed','avg_price','avg_price_f1','avg_price_f2','avg_profit_f1','avg_profit_f2','share_state_0.5_0.5','regime','mode_pair','share_mode_pair']
    print(top[cols].to_string(index=False))

    print(f"\nSaved per-run summary to: {args.csv}")


if __name__ == '__main__':
    main()

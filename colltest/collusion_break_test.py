#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Two-stage collusion-break test: Q-learning vs Q-learning pretraining, then intruder enters.

Stage A (pretraining):
  - Firm 1: Q-learning (1-step memory)
  - Firm 2: Q-learning (1-step memory)
  Run for T_train periods to (potentially) learn a collusive fixed-price outcome.

Stage B (intrusion test):
  - Firm 1: uses the Q-table learned in Stage A (optionally keeps learning)
  - Firm 2: replaced by an intruder algorithm (MBR / EXP3 / RBF)
  Run for T_test periods and measure whether the previously learned collusive outcome persists.

This script is designed for quick command-line experiments with smaller T and n.
It can optionally export a (downsampled) time series for Stage B.

Example quick run:
  python collusion_break_test.py --intruder mbr --T_train 50000 --T_test 50000 --n_seeds 20 --csv out_break_mbr.csv

Intruders:
  - mbr  : myopic best response (maximizes current-period profit)
  - exp3 : stateless EXP3 bandit over the price grid
  - rbf  : rule-based follower (matches Firm 1's current price)

Outputs:
  1) Summary CSV (one row per seed) containing Stage A/B late-phase averages and a simple "collusion_break" flag.
  2) Optional time-series CSV for Stage B (downsampled by --ts_stride).
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


def choose_argmax(rng: np.random.Generator, Qrow: np.ndarray, tie_break: str) -> int:
    if tie_break == 'first':
        return int(np.argmax(Qrow))
    m = Qrow.max()
    best = np.flatnonzero(Qrow == m)
    return int(rng.choice(best))


# ---------------- Intruders ----------------

def pick_mbr(rng: np.random.Generator, prices: np.ndarray, opp_idx: int) -> int:
    n = len(prices)
    vals = np.array([profit_from_indices(prices, a, opp_idx) for a in range(n)])
    best = np.flatnonzero(vals == vals.max())
    return int(rng.choice(best))


def exp3_init(n: int):
    return np.ones(n, dtype=float)


def exp3_probs(weights: np.ndarray, gamma: float):
    n = len(weights)
    w = weights / weights.sum()
    return (1.0 - gamma) * w + gamma / n


def exp3_sample(rng: np.random.Generator, weights: np.ndarray, gamma: float):
    probs = exp3_probs(weights, gamma)
    a = int(rng.choice(len(weights), p=probs))
    return a, probs


def exp3_update(weights: np.ndarray, a: int, reward: float, probs: np.ndarray, gamma: float, reward_scale: float):
    n = len(weights)
    rhat = max(0.0, min(1.0, reward / reward_scale))
    xhat = rhat / max(1e-12, probs[a])
    weights[a] *= math.exp((gamma * xhat) / n)


# ---------------- Core simulators ----------------

def run_stage_A_QQ(rng: np.random.Generator, T: int, prices: np.ndarray, alpha: float, delta: float,
                   tie_break: str, eps_floor: float = 0.0):
    """Stage A: Q-learning vs Q-learning (both 1-step memory). Returns Q1,Q2 and histories."""
    n = len(prices)

    # Exploration schedule
    theta = 1.0 - (1e-6) ** (1.0 / T)

    Q1 = np.zeros((n, n))
    Q2 = np.zeros((n, n))

    p1 = int(rng.integers(n))
    p2 = int(rng.integers(n))

    pend1 = None  # (s, a, imm)
    pend2 = None

    hist_p1 = np.empty(T)
    hist_p2 = np.empty(T)
    hist_pi1 = np.empty(T)
    hist_pi2 = np.empty(T)

    for t in range(T):
        eps = max(eps_floor, (1.0 - theta) ** t)

        if t % 2 == 0:
            # Firm 1 moves
            s = p2
            a = int(rng.integers(n)) if rng.random() < eps else choose_argmax(rng, Q1[s], tie_break)
            imm1 = profit_from_indices(prices, a, p2)

            # Update firm 2 from previous step (correct timing)
            if pend2 is not None:
                s2, a2, imm2 = pend2
                next_profit2 = profit_from_indices(prices, a2, a)
                best_future = Q2[a].max()
                target = imm2 + delta * next_profit2 + (delta ** 2) * best_future
                Q2[s2, a2] = (1.0 - alpha) * Q2[s2, a2] + alpha * target
                pend2 = None

            pend1 = (s, a, imm1)
            p1 = a

            pi1_now = imm1
            pi2_now = profit_from_indices(prices, p2, p1)

        else:
            # Firm 2 moves
            s = p1
            a = int(rng.integers(n)) if rng.random() < eps else choose_argmax(rng, Q2[s], tie_break)
            imm2 = profit_from_indices(prices, a, p1)

            # Update firm 1 from previous step (correct timing)
            if pend1 is not None:
                s1, a1, imm1 = pend1
                next_profit1 = profit_from_indices(prices, a1, a)
                best_future = Q1[a].max()
                target = imm1 + delta * next_profit1 + (delta ** 2) * best_future
                Q1[s1, a1] = (1.0 - alpha) * Q1[s1, a1] + alpha * target
                pend1 = None

            pend2 = (s, a, imm2)
            p2 = a

            pi1_now = profit_from_indices(prices, p1, p2)
            pi2_now = imm2

        hist_p1[t] = prices[p1]
        hist_p2[t] = prices[p2]
        hist_pi1[t] = pi1_now
        hist_pi2[t] = pi2_now

    return Q1, Q2, p1, p2, hist_p1, hist_p2, hist_pi1, hist_pi2


def run_stage_B_Q_vs_intruder(rng: np.random.Generator, T: int, prices: np.ndarray, Q1: np.ndarray,
                              alpha: float, delta: float, tie_break: str,
                              intruder: str, exp3_gamma: float,
                              q_continue_learning: bool,
                              q_eps: float,
                              ts_stride: int = 1):
    """Stage B: Firm1 uses learned Q1, Firm2 is intruder. Optionally continue learning Q1.

    Returns histories (possibly downsampled) and per-period profits arrays (full length for stats).
    """
    n = len(prices)

    # EXP3 state (if used)
    exp3_w = exp3_init(n) if intruder == 'exp3' else None
    reward_scale = 0.25

    # Start from random initial prices (or could be passed in; kept simple)
    p1 = int(rng.integers(n))
    p2 = int(rng.integers(n))

    pend1 = None

    # Full histories for profit stats
    hist_p1 = np.empty(T)
    hist_p2 = np.empty(T)
    hist_pi1 = np.empty(T)
    hist_pi2 = np.empty(T)

    # Downsampled time series for exporting
    ts_rows = []

    for t in range(T):
        if t % 2 == 0:
            # Firm 1 moves using learned Q1 (epsilon = q_eps)
            s1 = p2
            a1 = int(rng.integers(n)) if rng.random() < q_eps else choose_argmax(rng, Q1[s1], tie_break)
            imm1 = profit_from_indices(prices, a1, p2)

            pend1 = (s1, a1, imm1)
            p1 = a1

            pi1_now = imm1
            pi2_now = profit_from_indices(prices, p2, p1)

        else:
            # Intruder moves
            if intruder == 'mbr':
                a2 = pick_mbr(rng, prices, p1)
                probs = None
            elif intruder == 'rbf':
                a2 = int(p1)  # match
                probs = None
            elif intruder == 'exp3':
                a2, probs = exp3_sample(rng, exp3_w, exp3_gamma)
            else:
                raise ValueError(f'Unknown intruder: {intruder}')

            imm2 = profit_from_indices(prices, a2, p1)

            # Update Firm 1 from pending
            if pend1 is not None:
                s1, a1_prev, imm1_prev = pend1
                next_profit1 = profit_from_indices(prices, a1_prev, a2)
                best_future = Q1[a2].max()
                target = imm1_prev + delta * next_profit1 + (delta ** 2) * best_future
                if q_continue_learning:
                    Q1[s1, a1_prev] = (1.0 - alpha) * Q1[s1, a1_prev] + alpha * target
                pend1 = None

            if intruder == 'exp3':
                exp3_update(exp3_w, a2, imm2, probs, exp3_gamma, reward_scale)

            p2 = a2

            pi1_now = profit_from_indices(prices, p1, p2)
            pi2_now = imm2

        hist_p1[t] = prices[p1]
        hist_p2[t] = prices[p2]
        hist_pi1[t] = pi1_now
        hist_pi2[t] = pi2_now

        if ts_stride > 0 and (t % ts_stride == 0):
            ts_rows.append({'t': t, 'p1': float(prices[p1]), 'p2': float(prices[p2]), 'pi1': float(pi1_now), 'pi2': float(pi2_now)})

    ts_df = pd.DataFrame(ts_rows)
    return Q1, hist_p1, hist_p2, hist_pi1, hist_pi2, ts_df


def late_phase_metrics(prices: np.ndarray, hist_p1: np.ndarray, hist_p2: np.ndarray,
                       hist_pi1: np.ndarray, hist_pi2: np.ndarray, k: int):
    T = len(hist_p1)
    tail = int(0.1 * T)
    p1_last = hist_p1[-tail:]
    p2_last = hist_p2[-tail:]
    market_last = 0.5 * (p1_last + p2_last)

    mono_price = float(prices[int(np.argmin(np.abs(prices - 0.5)))])
    share_state_05_05 = float(np.mean((p1_last == mono_price) & (p2_last == mono_price)))

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

    return {
        'avg_price': float(np.mean(market_last)),
        'avg_price_f1': float(np.mean(p1_last)),
        'avg_price_f2': float(np.mean(p2_last)),
        'avg_profit_f1': float(np.mean(hist_pi1[-tail:])),
        'avg_profit_f2': float(np.mean(hist_pi2[-tail:])),
        'share_state_0.5_0.5': share_state_05_05,
        'mode_pair': str(mode_pair),
        'share_mode_pair': share_mode_pair,
        'std_market': std_market,
        'cycle_amp': cycle_amp,
        'jump_count_large': jump_count_large,
        'regime': regime,
    }


def self_play_two_stage(seed: int, T_train: int, T_test: int, k: int, alpha: float, delta: float,
                        intruder: str, exp3_gamma: float, tie_break: str,
                        q_continue_learning: bool, q_eps_test: float,
                        break_share_threshold: float, break_price_threshold: float,
                        save_ts: bool, ts_stride: int, ts_prefix: str) -> tuple[dict, pd.DataFrame | None]:

    rng = np.random.default_rng(seed)
    prices = np.linspace(0.0, 1.0, k + 1)

    # Stage A
    Q1, Q2, p1_end, p2_end, hp1A, hp2A, hpi1A, hpi2A = run_stage_A_QQ(
        rng, T_train, prices, alpha, delta, tie_break
    )
    A = late_phase_metrics(prices, hp1A, hp2A, hpi1A, hpi2A, k)

    # Stage B (intruder enters)
    Q1B, hp1B, hp2B, hpi1B, hpi2B, ts_df = run_stage_B_Q_vs_intruder(
        rng, T_test, prices, Q1, alpha, delta, tie_break,
        intruder, exp3_gamma, q_continue_learning, q_eps_test, ts_stride
    )
    B = late_phase_metrics(prices, hp1B, hp2B, hpi1B, hpi2B, k)

    # Simple collusion-break indicator:
    # If Stage A looks like monopoly (high share_state_0.5_0.5) but Stage B share or avg_price drops below thresholds.
    collusion_A = (A['share_state_0.5_0.5'] >= break_share_threshold) and (A['avg_price'] >= break_price_threshold)
    collusion_B = (B['share_state_0.5_0.5'] >= break_share_threshold) and (B['avg_price'] >= break_price_threshold)
    collusion_break = bool(collusion_A and (not collusion_B))

    row = {
        'seed': seed,
        'T_train': T_train,
        'T_test': T_test,
        'k': k,
        'alpha': alpha,
        'delta': delta,
        'intruder': intruder,
        'q_continue_learning': int(q_continue_learning),
        'q_eps_test': q_eps_test,
        # Stage A
        'A_avg_price': A['avg_price'],
        'A_avg_profit_f1': A['avg_profit_f1'],
        'A_avg_profit_f2': A['avg_profit_f2'],
        'A_share_state_0.5_0.5': A['share_state_0.5_0.5'],
        'A_regime': A['regime'],
        # Stage B
        'B_avg_price': B['avg_price'],
        'B_avg_profit_f1': B['avg_profit_f1'],
        'B_avg_profit_f2': B['avg_profit_f2'],
        'B_share_state_0.5_0.5': B['share_state_0.5_0.5'],
        'B_regime': B['regime'],
        # Break flag
        'collusion_break': int(collusion_break),
    }

    ts_out = None
    if save_ts:
        ts_df = ts_df.copy()
        ts_df['seed'] = seed
        ts_df['intruder'] = intruder
        ts_out = ts_df

    return row, ts_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--intruder', type=str, required=True, choices=['mbr','exp3','rbf'])

    ap.add_argument('--T_train', type=int, default=100_000)
    ap.add_argument('--T_test', type=int, default=100_000)
    ap.add_argument('--k', type=int, default=6)
    ap.add_argument('--alpha', type=float, default=0.3)
    ap.add_argument('--delta', type=float, default=0.95)

    ap.add_argument('--n_seeds', type=int, default=50)
    ap.add_argument('--seed_start', type=int, default=1)

    ap.add_argument('--exp3_gamma', type=float, default=0.07)
    ap.add_argument('--tie_break', type=str, default='random', choices=['random','first'])

    ap.add_argument('--q_continue_learning', action='store_true', help='If set, Firm 1 keeps updating Q in Stage B.')
    ap.add_argument('--q_eps_test', type=float, default=0.0, help='Exploration rate for Firm 1 in Stage B (default 0 = greedy).')

    ap.add_argument('--break_share_threshold', type=float, default=0.8, help='Threshold for share_state_0.5_0.5 to call monopoly-like.')
    ap.add_argument('--break_price_threshold', type=float, default=0.49, help='Threshold for avg_price to call monopoly-like.')

    ap.add_argument('--csv', type=str, default='results_break_test.csv')

    ap.add_argument('--save_ts', action='store_true', help='If set, saves a time-series CSV for Stage B.')
    ap.add_argument('--ts_stride', type=int, default=100, help='Record every ts_stride steps in Stage B time series.')
    ap.add_argument('--ts_prefix', type=str, default='ts_break', help='Prefix for time-series output file.')

    args = ap.parse_args()

    rows = []
    ts_frames = []

    seeds = range(args.seed_start, args.seed_start + args.n_seeds)
    for i, s in enumerate(seeds, start=1):
        row, ts_df = self_play_two_stage(
            seed=s,
            T_train=args.T_train,
            T_test=args.T_test,
            k=args.k,
            alpha=args.alpha,
            delta=args.delta,
            intruder=args.intruder,
            exp3_gamma=args.exp3_gamma,
            tie_break=args.tie_break,
            q_continue_learning=args.q_continue_learning,
            q_eps_test=args.q_eps_test,
            break_share_threshold=args.break_share_threshold,
            break_price_threshold=args.break_price_threshold,
            save_ts=args.save_ts,
            ts_stride=max(1, args.ts_stride),
            ts_prefix=args.ts_prefix,
        )
        rows.append(row)
        if ts_df is not None:
            ts_frames.append(ts_df)

        if i % max(1, args.n_seeds // 10) == 0:
            print(f'progress {i}/{args.n_seeds}')

    out = pd.DataFrame(rows)
    out.to_csv(args.csv, index=False)

    print('\n=== TWO-STAGE SUMMARY (pilot) ===')
    print(f"runs: {len(out)} | intruder={args.intruder} | T_train={args.T_train} | T_test={args.T_test} | k={args.k}")
    print(f"share collusion_break: {out['collusion_break'].mean():.2%}")
    print(f"A: mean avg_price={out['A_avg_price'].mean():.4f} | mean share_0.5_0.5={out['A_share_state_0.5_0.5'].mean():.2%}")
    print(f"B: mean avg_price={out['B_avg_price'].mean():.4f} | mean share_0.5_0.5={out['B_share_state_0.5_0.5'].mean():.2%}")
    print(f"Saved summary to: {args.csv}")

    if args.save_ts and ts_frames:
        ts_all = pd.concat(ts_frames, ignore_index=True)
        ts_path = f"{args.ts_prefix}_{args.intruder}.csv"
        ts_all.to_csv(ts_path, index=False)
        print(f"Saved Stage-B time series to: {ts_path} (stride={args.ts_stride})")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Self-play Q-learning under sequential pricing (Klein-style).
MODIFIED VERSION — ADDED FULL TIME SERIES EXPORT + COLLUSION LABEL

Functionality added:
--------------------
1. For every seed, saves a full time-series CSV:
       t, p1, p2, pi1, pi2, market_price, collusion_flag

2. Adds collusion_label to the summary CSV:
       "full" | "partial" | "none"
"""

import argparse
import numpy as np
import pandas as pd
import os


def demand(p_i: float, p_j: float) -> float:
    if p_i < p_j:
        return 1.0 - p_i
    if p_i == p_j:
        return 0.5 * (1.0 - p_i)
    return 0.0


def profit(prices: np.ndarray, i_idx: int, j_idx: int) -> float:
    p_i = float(prices[i_idx])
    p_j = float(prices[j_idx])
    return p_i * demand(p_i, p_j)


def self_play_one(seed: int, T: int, k: int, alpha: float, delta: float,
                  ts_dir: str = None,
                  tie_break: str = "random") -> dict:

    rng = np.random.default_rng(seed)
    prices = np.linspace(0.0, 1.0, k + 1)
    n = len(prices)

    theta = 1.0 - (1e-6)**(1.0 / T)

    Q1 = np.zeros((n, n))
    Q2 = np.zeros((n, n))

    p1 = int(rng.integers(n))
    p2 = int(rng.integers(n))

    pend1 = None
    pend2 = None

    log_p1 = np.empty(T)
    log_p2 = np.empty(T)
    log_pi1 = np.empty(T)
    log_pi2 = np.empty(T)

    def choose(Qrow: np.ndarray) -> int:
        if tie_break == "first":
            return int(np.argmax(Qrow))
        m = Qrow.max()
        best = np.flatnonzero(Qrow == m)
        return int(rng.choice(best))

    for t in range(T):

        eps = (1.0 - theta)**t

        if t % 2 == 0:
            # Firm 1 moves
            s = p2
            a = int(rng.integers(n)) if rng.random() < eps else choose(Q1[s])
            imm1 = profit(prices, a, p2)

            if pend2 is not None:
                s2, a2, imm2 = pend2
                next_profit2 = profit(prices, a2, a)
                best_future = Q2[a].max()
                target = imm2 + delta * next_profit2 + (delta**2) * best_future
                Q2[s2, a2] = (1-alpha)*Q2[s2, a2] + alpha * target
                pend2 = None

            pend1 = (s, a, imm1)
            p1 = a

            pi2_now = profit(prices, p2, p1)

            log_p1[t] = prices[p1]
            log_p2[t] = prices[p2]
            log_pi1[t] = imm1
            log_pi2[t] = pi2_now

        else:
            # Firm 2 moves
            s = p1
            a = int(rng.integers(n)) if rng.random() < eps else choose(Q2[s])
            imm2 = profit(prices, a, p1)

            if pend1 is not None:
                s1, a1, imm1 = pend1
                next_profit1 = profit(prices, a1, a)
                best_future = Q1[a].max()
                target = imm1 + delta * next_profit1 + (delta**2) * best_future
                Q1[s1, a1] = (1-alpha)*Q1[s1, a1] + alpha * target
                pend1 = None

            pend2 = (s, a, imm2)
            p2 = a

            pi1_now = profit(prices, p1, p2)

            log_p1[t] = prices[p1]
            log_p2[t] = prices[p2]
            log_pi1[t] = pi1_now
            log_pi2[t] = imm2

    # === Late-phase summary ===
    tail = int(0.1 * T)
    p1_last = log_p1[-tail:]
    p2_last = log_p2[-tail:]
    t_last = np.arange(T-tail, T)

    market_last = 0.5*(p1_last + p2_last)
    mono_price = float(prices[int(np.argmin(np.abs(prices - 0.5)))])

    share_state_05_05 = float(np.mean((p1_last == mono_price) &
                                      (p2_last == mono_price)))

    mask_f1 = (t_last % 2 == 0)
    idx_even = np.where(mask_f1)[0]
    idx_even = idx_even[idx_even+1 < len(t_last)]
    share_joint_consec_05 = float(np.mean(
        (p1_last[idx_even] == mono_price) &
        (p2_last[idx_even+1] == mono_price)
    )) if len(idx_even) else 0.0

    pair_series = pd.Series(list(zip(np.round(p1_last, 6),
                                     np.round(p2_last, 6))))
    mode_pair = pair_series.value_counts().idxmax()
    share_mode_pair = float(pair_series.value_counts(normalize=True).max())

    tick = 1.0/k
    diffs = np.diff(market_last)
    jump_count_large = int(np.sum(np.abs(diffs) > tick + 1e-12)) if len(diffs) else 0

    std_market = float(np.std(market_last))
    cycle_amp = float(np.max(market_last) - np.min(market_last))
    share_equal = float(np.mean(p1_last == p2_last))

    if share_state_05_05 > 0.8 and std_market < 0.02:
        regime = "monopoly_regime"
    elif share_equal > 0.8 and std_market < 0.02:
        regime = "focal_equal_price"
    elif cycle_amp > 0.15:
        regime = "cycling/asymmetric"
    else:
        regime = "mixed"

    avg_price = float(np.mean(market_last))
    avg_profit = float(0.5*(np.mean(log_pi1[-tail:]) +
                            np.mean(log_pi2[-tail:])))

    # === COLLUSION LABEL ===
    if share_state_05_05 > 0.95 and std_market < 0.02:
        collusion_label = "full"
    elif avg_price > 0.40 and share_state_05_05 > 0.10:
        collusion_label = "partial"
    else:
        collusion_label = "none"

    # === SAVE TIME-SERIES ===
    if ts_dir is not None:
        os.makedirs(ts_dir, exist_ok=True)
        df_ts = pd.DataFrame({
            "t": np.arange(T),
            "p1": log_p1,
            "p2": log_p2,
            "pi1": log_pi1,
            "pi2": log_pi2,
            "market_price": 0.5 * (log_p1 + log_p2),
            "collusion_flag": collusion_label,
        })
        df_ts.to_csv(f"{ts_dir}/ts_seed_{seed}.csv", index=False)

    # === RETURN SUMMARY ROW ===
    return dict(
        seed=seed,
        T=T,
        k=k,
        avg_price=avg_price,
        avg_profit=avg_profit,
        share_state_05_05=share_state_05_05,
        share_joint_consecutive_0_5=share_joint_consec_05,
        mode_pair=str(mode_pair),
        share_mode_pair=share_mode_pair,
        std_market=std_market,
        cycle_amp=cycle_amp,
        jump_count_large=jump_count_large,
        regime=regime,
        collusion_label=collusion_label
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=500000)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--delta", type=float, default=0.95)
    ap.add_argument("--n_seeds", type=int, default=1000)
    ap.add_argument("--seed_start", type=int, default=1)
    ap.add_argument("--csv", type=str, default="results.csv")
    ap.add_argument("--ts_dir", type=str, default=None,
                    help="Directory for saving per-seed time-series")

    args = ap.parse_args()

    rows = []
    seeds = range(args.seed_start, args.seed_start + args.n_seeds)

    for i, seed in enumerate(seeds, start=1):
        out = self_play_one(
            seed, args.T, args.k,
            args.alpha, args.delta,
            ts_dir=args.ts_dir
        )
        rows.append(out)

        if i % max(1, args.n_seeds // 10) == 0:
            print(f"progress {i}/{args.n_seeds}")

    df = pd.DataFrame(rows)
    df.to_csv(args.csv, index=False)

    print("\n=== AGGREGATE SUMMARY (late phase) ===")
    print(df.describe(include="all"))

    print(f"\nSaved per-run summary to: {args.csv}")
    if args.ts_dir:
        print(f"Time-series saved to directory: {args.ts_dir}")


if __name__ == "__main__":
    main()

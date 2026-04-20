#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q-learning (1-step memory) vs Q-learning with 2-period market memory.
(Firm 2 observes (opponent_last, own_last))

Run example:
  python self_play_2period_runner.py --T 500000 --n_seeds 1000 \\
         --csv results.csv --ts_dir ts --save_q_dir q_tables
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
                  ts_dir: str = None, save_q_dir: str = None,
                  tie_break: str = "random") -> dict:
    rng = np.random.default_rng(seed)
    prices = np.linspace(0.0, 1.0, k + 1)
    n = len(prices)

    theta = 1.0 - (1e-6) ** (1.0 / T)

    # Firm 1: state = opponent's last price
    Q1 = np.zeros((n, n))
    # Firm 2: state = (opponent_last, own_last)
    Q2 = np.zeros((n, n, n))

    # Initial prices (indices)
    p1 = int(rng.integers(n))
    p2 = int(rng.integers(n))

    pend1 = None  # (s1=p2, a1, imm1)
    pend2 = None  # (s_p1, s_p2, a2, imm2)

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
        eps = (1.0 - theta) ** t

        if t % 2 == 0:
            # ----- Firm 1 moves -----
            s1 = p2
            a1 = int(rng.integers(n)) if rng.random() < eps else choose(Q1[s1])
            imm1 = profit(prices, a1, p2)

            # Update Firm 2 from its previous move
            if pend2 is not None:
                s_p1, s_p2, a2, imm2 = pend2
                next_profit2 = profit(prices, a2, a1)   # firm2's price a2 vs new firm1 price a1
                best_future = Q2[a1, a2].max()          # next state = (a1, a2)
                target = imm2 + delta * next_profit2 + (delta ** 2) * best_future
                Q2[s_p1, s_p2, a2] = (1.0 - alpha) * Q2[s_p1, s_p2, a2] + alpha * target
                pend2 = None

            pend1 = (s1, a1, imm1)
            p1 = a1

            pi1_now = imm1
            pi2_now = profit(prices, p2, p1)

        else:
            # ----- Firm 2 moves -----
            s_p1, s_p2 = p1, p2
            a2 = int(rng.integers(n)) if rng.random() < eps else choose(Q2[s_p1, s_p2])
            imm2 = profit(prices, a2, p1)

            # Update Firm 1 from its previous move
            if pend1 is not None:
                s1, a1, imm1 = pend1
                next_profit1 = profit(prices, a1, a2)
                best_future = Q1[a2].max()
                target = imm1 + delta * next_profit1 + (delta ** 2) * best_future
                Q1[s1, a1] = (1.0 - alpha) * Q1[s1, a1] + alpha * target
                pend1 = None

            pend2 = (s_p1, s_p2, a2, imm2)
            p2 = a2

            pi1_now = profit(prices, p1, p2)
            pi2_now = imm2

        log_p1[t] = prices[p1]
        log_p2[t] = prices[p2]
        log_pi1[t] = pi1_now
        log_pi2[t] = pi2_now

    # ----- Late phase summary (last 10%) -----
    tail = int(0.1 * T)
    p1_last = log_p1[-tail:]
    p2_last = log_p2[-tail:]
    pi1_last = log_pi1[-tail:]
    pi2_last = log_pi2[-tail:]

    avg_price1 = float(np.mean(p1_last))
    avg_price2 = float(np.mean(p2_last))
    avg_profit1 = float(np.mean(pi1_last))
    avg_profit2 = float(np.mean(pi2_last))

    market_last = 0.5 * (p1_last + p2_last)
    tick = 1.0 / k
    diffs = np.diff(market_last)
    jump_count_large = int(np.sum(np.abs(diffs) > tick + 1e-12)) if len(diffs) else 0
    cycle_amp = float(np.max(market_last) - np.min(market_last))

    # Najczęstsza para cen
    pair_series = pd.Series(list(zip(np.round(p1_last, 6), np.round(p2_last, 6))))
    mode_pair = str(pair_series.value_counts().idxmax()) if not pair_series.empty else "(nan,nan)"

    # ----- Save time series -----
    if ts_dir is not None:
        os.makedirs(ts_dir, exist_ok=True)
        df_ts = pd.DataFrame({
            "t": np.arange(T),
            "p1": log_p1,
            "p2": log_p2,
            "pi1": log_pi1,
            "pi2": log_pi2,
            "market_price": 0.5 * (log_p1 + log_p2),
        })
        df_ts.to_parquet(
            f"{ts_dir}/ts_seed_{seed}.parquet",
            engine='pyarrow', compression='zstd', index=False
        )

    # ----- Save Q-tables -----
    if save_q_dir is not None:
        os.makedirs(save_q_dir, exist_ok=True)
        np.save(os.path.join(save_q_dir, f"Q1_seed_{seed}.npy"), Q1)
        np.save(os.path.join(save_q_dir, f"Q2_seed_{seed}.npy"), Q2)

    # Zwracamy tylko żądane kolumny
    return {
        "seed": seed,
        "T": T,
        "k": k,
        "avg_price1": avg_price1,
        "avg_price2": avg_price2,
        "avg_profit1": avg_profit1,
        "avg_profit2": avg_profit2,
        "mode_pair": mode_pair,
        "cycle_amp": cycle_amp,
        "jump_count_large": jump_count_large,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=500_000)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--delta", type=float, default=0.95)
    ap.add_argument("--n_seeds", type=int, default=1000)
    ap.add_argument("--seed_start", type=int, default=1)
    ap.add_argument("--csv", type=str, default="results_2period.csv")
    ap.add_argument("--ts_dir", type=str, default=None,
                    help="Directory for saving per-seed time-series (Parquet)")
    ap.add_argument("--save_q_dir", type=str, default=None,
                    help="Directory for saving final Q-tables as .npy files")
    args = ap.parse_args()

    seeds = range(args.seed_start, args.seed_start + args.n_seeds)
    rows = []
    for i, s in enumerate(seeds, start=1):
        rows.append(self_play_one(
            s, args.T, args.k, args.alpha, args.delta,
            ts_dir=args.ts_dir, save_q_dir=args.save_q_dir
        ))
        if i % max(1, args.n_seeds // 10) == 0:
            print(f"progress {i}/{args.n_seeds}")

    df = pd.DataFrame(rows)
    df.to_csv(args.csv, index=False)

    print("\n=== AGGREGATE SUMMARY (late phase) ===")
    print(f"runs: {len(df)} | T={args.T} | k={args.k} | alpha={args.alpha} | delta={args.delta}")
    print(f"avg(avg_price1): {df['avg_price1'].mean():.4f} | median: {df['avg_price1'].median():.4f}")
    print(f"avg(avg_price2): {df['avg_price2'].mean():.4f} | median: {df['avg_price2'].median():.4f}")
    print(f"avg(avg_profit1): {df['avg_profit1'].mean():.4f} | median: {df['avg_profit1'].median():.4f}")
    print(f"avg(avg_profit2): {df['avg_profit2'].mean():.4f} | median: {df['avg_profit2'].median():.4f}")
    print(f"mean cycle_amp: {df['cycle_amp'].mean():.4f}")
    print(f"mean jump_count_large: {df['jump_count_large'].mean():.2f}")

    top = df.sort_values('avg_profit1', ascending=False).head(10)
    print("\nTop 10 by avg_profit1:")
    cols = ['seed', 'avg_price1', 'avg_price2', 'avg_profit1', 'avg_profit2', 'cycle_amp', 'jump_count_large']
    print(top[cols].to_string(index=False))

    print(f"\nSaved per-run summary to: {args.csv}")
    if args.ts_dir:
        print(f"Time-series saved to: {args.ts_dir}")
    if args.save_q_dir:
        print(f"Q-tables saved to: {args.save_q_dir}")


if __name__ == "__main__":
    main()

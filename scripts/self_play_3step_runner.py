#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Q-learning (1-step memory) vs Q-learning (3-step memory), Klein-style sequential pricing.

Asymmetric learning setup
-------------------------
- Firm 1: tabular Q-learning with 1-step memory.
          State = opponent's last price index (p2).
- Firm 2: tabular Q-learning with 3-step memory.
          State = (opponent_last, opponent_prev1, opponent_prev2)
          i.e., the last three prices chosen by Firm 1 (from its last three moves).

Everything else matches the baseline runner:
- Discrete price grid {0, 1/k, ..., 1}
- Alternating moves (firm 1 at even t, firm 2 at odd t)
- Epsilon-greedy exploration with a decaying schedule
- Klein-style two-step target:  π_t + δ·π_{t+1} + δ^2·max Q(next_state, ·)
  with correct timing: π_{t+1} uses the agent's own price chosen at time t.

Output CSV includes per-firm late-phase averages:
- avg_price_f1, avg_price_f2
- avg_profit_f1, avg_profit_f2
"""

import argparse
import numpy as np
import pandas as pd


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


def self_play_one(
    seed: int,
    T: int,
    k: int,
    alpha: float,
    delta: float,
    tie_break: str = "random",
) -> dict:
    rng = np.random.default_rng(seed)

    prices = np.linspace(0.0, 1.0, k + 1)
    n = len(prices)

    # Exploration schedule (same structure as baseline)
    theta = 1.0 - (1e-6) ** (1.0 / T)

    # Firm 1: state = p2_last (n states), action = own price (n actions)
    Q1 = np.zeros((n, n))

    # Firm 2: state = (p1_last, p1_prev1, p1_prev2) (n^3 states), action = own price (n actions)
    Q2 = np.zeros((n, n, n, n))

    # Initial price indices
    p1 = int(rng.integers(n))
    p2 = int(rng.integers(n))

    # Firm 2 tracks the last three Firm-1 actions (from Firm-1's last three moves)
    p1_prev1 = p1  # one Firm-1 move back
    p1_prev2 = p1  # two Firm-1 moves back

    # Pending tuples
    # firm1: (state_idx=p2_last, action_idx=a1, immediate_profit)
    pend1 = None
    # firm2: (s_last, s_prev1, s_prev2, action_idx=a2, immediate_profit)
    pend2 = None

    hist_p1 = np.empty(T)
    hist_p2 = np.empty(T)
    hist_pi1 = np.empty(T)
    hist_pi2 = np.empty(T)

    def choose(Qrow: np.ndarray) -> int:
        if tie_break == "first":
            return int(np.argmax(Qrow))
        m = Qrow.max()
        best = np.flatnonzero(Qrow == m)
        return int(rng.choice(best))

    for t in range(T):
        eps = (1.0 - theta) ** t

        if t % 2 == 0:
            # -------- Firm 1 moves --------
            # cache Firm-1 history BEFORE updating (needed for Firm 2 next-state)
            p1_old = p1
            p1_prev1_old = p1_prev1
            p1_prev2_old = p1_prev2

            s1 = p2
            a1 = int(rng.integers(n)) if rng.random() < eps else choose(Q1[s1])
            imm1 = profit(prices, a1, p2)

            # Update Firm 2 from its previous move (if any)
            if pend2 is not None:
                s2_last, s2_prev1, s2_prev2, a2, imm2 = pend2

                # Next-period profit for Firm 2 uses its own price a2 vs new Firm-1 price a1
                next_profit2 = profit(prices, a2, a1)

                # Next state for Firm 2 after Firm 1 moves:
                # (opp_last=a1, opp_prev1=p1_old, opp_prev2=p1_prev1_old)
                best_future = Q2[a1, p1_old, p1_prev1_old].max()

                target = imm2 + delta * next_profit2 + (delta ** 2) * best_future
                Q2[s2_last, s2_prev1, s2_prev2, a2] = (1.0 - alpha) * Q2[s2_last, s2_prev1, s2_prev2, a2] + alpha * target

                pend2 = None

            # set pending for Firm 1
            pend1 = (s1, a1, imm1)

            # update Firm 1 price and shift Firm-1 history for Firm 2
            p1 = a1
            p1_prev2 = p1_prev1_old
            p1_prev1 = p1_old

            # record profits this period
            pi1_now = imm1
            pi2_now = profit(prices, p2, p1)

        else:
            # -------- Firm 2 moves --------
            s2_last = p1
            s2_prev1 = p1_prev1
            s2_prev2 = p1_prev2

            a2 = int(rng.integers(n)) if rng.random() < eps else choose(Q2[s2_last, s2_prev1, s2_prev2])
            imm2 = profit(prices, a2, p1)

            # Update Firm 1 from its previous move (if any)
            if pend1 is not None:
                s1, a1, imm1 = pend1

                # Next-period profit for Firm 1 uses its own price a1 vs new Firm-2 price a2
                next_profit1 = profit(prices, a1, a2)

                # Next state for Firm 1 after Firm 2 moves is opponent_last=a2
                best_future = Q1[a2].max()

                target = imm1 + delta * next_profit1 + (delta ** 2) * best_future
                Q1[s1, a1] = (1.0 - alpha) * Q1[s1, a1] + alpha * target

                pend1 = None

            # set pending for Firm 2
            pend2 = (s2_last, s2_prev1, s2_prev2, a2, imm2)

            # update Firm 2 price
            p2 = a2

            # record profits this period
            pi1_now = profit(prices, p1, p2)
            pi2_now = imm2

        # record histories
        hist_p1[t] = prices[p1]
        hist_p2[t] = prices[p2]
        hist_pi1[t] = pi1_now
        hist_pi2[t] = pi2_now

    # ---------------- Late phase summary ----------------
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
        if len(idx_even)
        else 0.0
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
        regime = "monopoly_regime"
    elif share_equal > 0.8 and std_market < 0.02:
        regime = "focal_equal_price"
    elif cycle_amp > 0.15:
        regime = "cycling/asymmetric"
    else:
        regime = "mixed"

    # Market averages
    avg_price = float(np.mean(market_last))

    # Per-firm late-phase averages
    avg_price_f1 = float(np.mean(p1_last))
    avg_price_f2 = float(np.mean(p2_last))
    avg_profit_f1 = float(np.mean(hist_pi1[-tail:]))
    avg_profit_f2 = float(np.mean(hist_pi2[-tail:]))

    avg_profit = float(0.5 * (avg_profit_f1 + avg_profit_f2))

    return {
        "seed": seed,
        "T": T,
        "k": k,
        "avg_price": avg_price,
        "avg_price_f1": avg_price_f1,
        "avg_price_f2": avg_price_f2,
        "avg_profit": avg_profit,
        "avg_profit_f1": avg_profit_f1,
        "avg_profit_f2": avg_profit_f2,
        "share_state_0.5_0.5": share_state_05_05,
        "share_joint_consecutive_0.5": share_joint_consecutive_05,
        "mode_pair": str(mode_pair),
        "share_mode_pair": share_mode_pair,
        "std_market": std_market,
        "cycle_amp": cycle_amp,
        "jump_count_large": jump_count_large,
        "regime": regime,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=100_000)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--delta", type=float, default=0.95)
    ap.add_argument("--n_seeds", type=int, default=50)
    ap.add_argument("--seed_start", type=int, default=1)
    ap.add_argument("--csv", type=str, default="results_3step.csv")
    args = ap.parse_args()

    seeds = range(args.seed_start, args.seed_start + args.n_seeds)
    rows = []

    for i, s in enumerate(seeds, start=1):
        rows.append(self_play_one(s, args.T, args.k, args.alpha, args.delta))
        if i % max(1, args.n_seeds // 10) == 0:
            print(f"progress {i}/{args.n_seeds}")

    df = pd.DataFrame(rows)
    df.to_csv(args.csv, index=False)

    print("\n=== AGGREGATE SUMMARY (late phase) ===")
    print(f"runs: {len(df)} | T={args.T} | k={args.k} | alpha={args.alpha} | delta={args.delta}")

    print(f"avg(avg_price): {df['avg_price'].mean():.4f} | median: {df['avg_price'].median():.4f}")
    print(f"avg(avg_profit): {df['avg_profit'].mean():.4f} | median: {df['avg_profit'].median():.4f}")

    print(f"avg(avg_profit_f1): {df['avg_profit_f1'].mean():.4f} | median: {df['avg_profit_f1'].median():.4f}")
    print(f"avg(avg_profit_f2): {df['avg_profit_f2'].mean():.4f} | median: {df['avg_profit_f2'].median():.4f}")
    print(f"avg(avg_price_f1):  {df['avg_price_f1'].mean():.4f} | median: {df['avg_price_f1'].median():.4f}")
    print(f"avg(avg_price_f2):  {df['avg_price_f2'].mean():.4f} | median: {df['avg_price_f2'].median():.4f}")
    print(f"share firm2_profit>firm1_profit: {(df['avg_profit_f2'] > df['avg_profit_f1']).mean():.2%}")

    print(f"share monopoly_regime: {(df['regime']=='monopoly_regime').mean():.2%}")
    print(f"mean share_state_0.5_0.5: {df['share_state_0.5_0.5'].mean():.2%}")

    top = df.sort_values('share_state_0.5_0.5', ascending=False).head(10)
    print("\nTop 10 by share_state_0.5_0.5:")
    cols = ['seed','avg_price','avg_price_f1','avg_price_f2','avg_profit','avg_profit_f1','avg_profit_f2',
            'share_state_0.5_0.5','regime','mode_pair','share_mode_pair']
    print(top[cols].to_string(index=False))

    print(f"\nSaved per-run summary to: {args.csv}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generator danych do wykresów Figure 1 z artykułu Kleina:
- Średnia optymalność (Gamma) w funkcji T
- Udział równowag Nasha w funkcji T

Użycie:
    python generate_figure1_data.py --T_list "0,25000,50000,..." --n_seeds 1000 --parallel
"""

import argparse
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

# ===========================
# Funkcja symulacyjna (z poprzedniego kodu, z drobnymi modyfikacjami)
# ===========================
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
                  nash_tolerance: float = 1e-5) -> dict:
    rng = np.random.default_rng(seed)
    prices = np.linspace(0.0, 1.0, k + 1)
    n = len(prices)

    theta = 1.0 - (1e-6) ** (1.0 / T)

    Q1 = np.zeros((n, n))
    Q2 = np.zeros((n, n))

    p1 = int(rng.integers(n))
    p2 = int(rng.integers(n))

    pend1 = None
    pend2 = None

    def choose(Qrow: np.ndarray) -> int:
        m = Qrow.max()
        best = np.flatnonzero(Qrow == m)
        return int(rng.choice(best))

    for t in range(T):
        eps = (1.0 - theta) ** t

        if t % 2 == 0:  # firma 1
            s = p2
            a = int(rng.integers(n)) if rng.random() < eps else choose(Q1[s])
            imm1 = profit(prices, a, p2)

            if pend2 is not None:
                s2, a2, imm2 = pend2
                next_profit2 = profit(prices, a2, a)
                best_future = Q2[a].max()
                target = imm2 + delta * next_profit2 + (delta**2) * best_future
                Q2[s2, a2] = (1 - alpha) * Q2[s2, a2] + alpha * target
                pend2 = None

            pend1 = (s, a, imm1)
            p1 = a

        else:  # firma 2
            s = p1
            a = int(rng.integers(n)) if rng.random() < eps else choose(Q2[s])
            imm2 = profit(prices, a, p1)

            if pend1 is not None:
                s1, a1, imm1 = pend1
                next_profit1 = profit(prices, a1, a)
                best_future = Q1[a].max()
                target = imm1 + delta * next_profit1 + (delta**2) * best_future
                Q1[s1, a1] = (1 - alpha) * Q1[s1, a1] + alpha * target
                pend1 = None

            pend2 = (s, a, imm2)
            p2 = a

    # Obliczenia końcowe
    last_p1_idx = int(np.argmin(np.abs(prices - prices[p1])))
    last_p2_idx = int(np.argmin(np.abs(prices - prices[p2])))

    max_q1_state = Q1[last_p2_idx].max()
    gamma1 = Q1[last_p2_idx, last_p1_idx] / max_q1_state if max_q1_state != 0 else np.nan

    max_q2_state = Q2[last_p1_idx].max()
    gamma2 = Q2[last_p1_idx, last_p2_idx] / max_q2_state if max_q2_state != 0 else np.nan

    nash = bool(abs(gamma1 - 1.0) < nash_tolerance and abs(gamma2 - 1.0) < nash_tolerance)

    return {'gamma1': gamma1, 'gamma2': gamma2, 'nash': nash}

# ===========================
# Funkcja pomocnicza do równoległego uruchamiania
# ===========================
def run_T(args):
    T, k, alpha, delta, n_seeds, seed_start, nash_tolerance = args
    print(f"  T = {T} ...")
    gamma_all = []
    nash_count = 0
    for seed in range(seed_start, seed_start + n_seeds):
        out = self_play_one(seed, T, k, alpha, delta, nash_tolerance)
        gamma_all.append(out['gamma1'])
        gamma_all.append(out['gamma2'])
        if out['nash']:
            nash_count += 1
    avg_gamma = np.nanmean(gamma_all) if gamma_all else np.nan
    share_nash = nash_count / n_seeds
    print(f"  T = {T} gotowe: gamma = {avg_gamma:.3f}, Nash = {share_nash:.3f}")
    return T, avg_gamma, share_nash

# ===========================
# Główna funkcja
# ===========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T_list", type=str,
                    default="0,25000,50000,75000,100000,125000,150000,175000,200000,225000,250000,275000,300000,325000,350000,375000,400000,425000,450000,475000,500000",
                    help="Lista wartości T oddzielonych przecinkami")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--delta", type=float, default=0.95)
    ap.add_argument("--n_seeds", type=int, default=1000)
    ap.add_argument("--seed_start", type=int, default=1)
    ap.add_argument("--nash_tolerance", type=float, default=1e-5)
    ap.add_argument("--output", type=str, default="figure1_data.csv")
    ap.add_argument("--parallel", action="store_true", help="Uruchom równolegle dla różnych T")
    args = ap.parse_args()

    T_values = [int(x.strip()) for x in args.T_list.split(",")]
    print(f"Generowanie danych dla {len(T_values)} wartości T, {args.n_seeds} seedów na T")

    results = []
    if args.parallel:
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(run_T, (T, args.k, args.alpha, args.delta, args.n_seeds, args.seed_start, args.nash_tolerance))
                for T in T_values
            ]
            for future in as_completed(futures):
                results.append(future.result())
    else:
        for T in T_values:
            results.append(run_T((T, args.k, args.alpha, args.delta, args.n_seeds, args.seed_start, args.nash_tolerance)))

    # Sortowanie po T
    results.sort(key=lambda x: x[0])
    df = pd.DataFrame(results, columns=["T", "avg_gamma", "share_nash"])
    df.to_csv(args.output, index=False)
    print(f"\nZapisano do pliku: {args.output}")
    print(df)

if __name__ == "__main__":
    main()

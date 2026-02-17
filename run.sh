#!/bin/bash

python self_play_2period_runner.py --T 500000 --n_seeds 1000 --csv 2period.csv
python self_play_3period_runner.py --T 500000 --n_seeds 1000 --csv 3period.csv
python q_vs_intruder.py --intruder mbr  --T 500000 --n_seeds 1000 --csv mbr.csv
python q_vs_intruder.py --intruder exp3 --T 500000 --n_seeds 1000 --csv exp3.csv
python q_vs_intruder.py --intruder rbf  --T 500000 --n_seeds 1000 --csv rbf.csv
python collusion_break_test.py --intruder mbr --T_train 500000 --T_test 500000 --n_seeds 1000 --csv break_mbr.csv
python collusion_break_test.py --intruder exp3 --T_train 500000 --T_test 500000 --n_seeds 1000 --csv break_exp3.csv
python collusion_break_test.py --intruder rbf --T_train 500000 --T_test 500000 --n_seeds 1000 --csv break_rbf.csv

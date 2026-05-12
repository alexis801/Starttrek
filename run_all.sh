#!/bin/bash
set -e

python3 src/model_train/train_dqn.py --config configs/train_dqn.yaml
python3 src/model_train/train_double_dqn.py --config configs/train_double_dqn.yaml

# eval DQN
python src/eval.py --model models/model_dqn.pth --csv-out stats/csv/eval_dqn.csv

# eval Double DQN
python src/eval.py --model models/model_double_dqn.pth --csv-out stats/csv/eval_double_dqn.csv

echo "END"
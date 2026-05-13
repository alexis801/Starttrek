#!/bin/bash
set -e

#train DQN
echo "Training DQN..."
python3 src/model_train/train_dqn.py

#train double_dqn
echo "Training Double_DQN"
python3 src/model_train/train_double_dqn.py

# eval DQN
echo "Evaluating DQN..."
python3 src/eval.py --model models/model_dqn.pth --csv-out stats/csv/eval_dqn.csv

# eval Double DQN
echo "Evaluating Double DQN..."
python3 src/eval.py --model models/model_double_dqn.pth --csv-out stats/csv/eval_double_dqn.csv

echo "Generating plots..."
mkdir -p stats/plots/dqn stats/plots/double_dqn
python3 stats/plots/plot.py stats/csv/eval_dqn.csv  stats/plots/dqn/
python3 stats/plots/plot.py stats/csv/eval_double_dqn.csv  stats/plots/double_dqn/

echo "END"
##
## EPITECH PROJECT, 2026
## starttrek
## File description:
## save_stat
##
from pathlib import Path
import matplotlib.pyplot as plt
import os
import numpy as np
import csv

#path to root from the current file
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))

#Plot that save loss, reward and epsilon
STATS_DIR = Path(PROJECT_ROOT) / "stats"
PLOTS_DIR = STATS_DIR / "plots"
CSV_DIR = STATS_DIR / "csv"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)

def moving_average(data, window=100):
    return np.convolve(data, np.ones(window) / window, mode='valid')

def plot_rewards(rewards_history):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_history, label="Reward")

    if len(rewards_history) >= 100:
        ma = moving_average(rewards_history, 100)
        plt.plot(range(99, len(rewards_history)), ma, label="Average")

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training rewards")
    plt.legend()
    path = PLOTS_DIR / "rewards.png"
    plt.savefig(path)
    plt.close()
    print(f"Saved reward plot to '{path}'")


def plot_epsilon(epsilon_history):
    plt.figure(figsize=(10, 5))
    plt.plot(epsilon_history)
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.title("Epsilon stat")
    path = PLOTS_DIR / "epsilon.png"
    plt.savefig(path)
    plt.close()
    print(f"Saved epsilon plot to '{path}'")


def plot_loss(loss_history):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    path = PLOTS_DIR / "loss.png"
    plt.savefig(path)
    plt.close()
    print(f"Saved loss plot to '{path}'")

def save_csv(rewards_history, epsilon_history, loss_history):
    output_path = CSV_DIR / "train_metrics.csv"

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "epsilon", "loss"])

        for i in range(len(rewards_history)):
            writer.writerow([i + 1, rewards_history[i],
                epsilon_history[i], loss_history[i]])
    print(f"Saved csv to '{output_path}'")
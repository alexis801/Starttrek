##
## EPITECH PROJECT, 2025
## star-trek
## File description:
## plot.py
##

#generate plots from the csv files in stats/csv

import csv
import sys
import matplotlib.pyplot as plt

csv_path = sys.argv[1]
output_path = sys.argv[2]

episodes = []
rewards = []
terminated = []
truncated = []

with (open(csv_path) as file):
    for row in csv.DictReader(file):
        episodes.append(int(row["episode"]))
        rewards.append(float(row["reward"]))
        terminated.append(int(row["terminated"]))
        truncated.append(int(row["truncated"]))

plt.plot(episodes, rewards)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Reward per Episode")
plt.savefig(f"{output_path}/rewards.png")
plt.close()

plt.bar(["Terminated", "Truncated"], [sum(terminated), sum(truncated)])
plt.title("Termination Causes")
plt.savefig(f"{output_path}/termination.png")
plt.close()

print(f"Plots saved to {output_path}")
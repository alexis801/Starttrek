import gymnasium as gym
import torch
import torch.nn as nn
import random
from collections import deque
import torch.optim as optim
import sys
import os
import numpy as np
from save_stat import plot_epsilon, plot_rewards, plot_loss, save_csv

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))

sys.path.insert(0, PROJECT_ROOT)

from utils.config import load_merged_config
from utils.seed import set_seed

# Config default and utilities for loading and merging YAML configs
CONFIG_DEFAULT = os.path.join(PROJECT_ROOT, "configs/default.yaml")
CONFIG_TRAIN = os.path.join(PROJECT_ROOT, "configs/train_double_dqn.yaml")

config = load_merged_config(CONFIG_DEFAULT, CONFIG_TRAIN)

#config
LR = config["hyperparameters"]["lr"]
GAMMA = config["hyperparameters"]["gamma"]
BATCH_SIZE = config["hyperparameters"]["batch_size"]
BUFFER_SIZE = config["hyperparameters"]["buffer_size"]

#EPS
EPS_START = config["exploration"]["epsilon_start"]
EPS_MIN = config["exploration"]["epsilon_min"]
EPS_DECAY = config["exploration"]["epsilon_decay"]

#training
NUM_EPISODES = config["training"]["num_episodes"]
WARMUP_STEPS = config["training"]["warmup_steps"]
TARGET_UPDATE_EVERY = config["training"]["target_update_every"]

#Import seed from the config file
SEED = config["seeds"]["base"]

class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )
    def forward(self, x):
        return self.net(x)

model = QNetwork()
target_model = QNetwork()
target_model.load_state_dict(model.state_dict())

optimizer = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.SmoothL1Loss()

buffer = deque(maxlen=BUFFER_SIZE)

def store(obs, action, reward, next_obs, done):
    buffer.append((obs, action, reward, next_obs, done))

def policy(obs, env, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()
    obs_t = torch.tensor(obs, dtype=torch.float32)
    return torch.argmax(model(obs_t)).item()

def train():
    if len(buffer) < BATCH_SIZE:
        return
    batch = random.sample(buffer, BATCH_SIZE)
    obs, actions, rewards, next_obs, dones = zip(*batch)

    obs = torch.tensor(np.array(obs), dtype=torch.float32)
    actions = torch.tensor(np.array(actions), dtype=torch.int64)
    rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
    next_obs = torch.tensor(np.array(next_obs), dtype=torch.float32)
    dones = torch.tensor(np.array(dones), dtype=torch.float32)

    q = model(obs).gather(1, actions.unsqueeze(1)).squeeze()
    next_actions = model(next_obs).argmax(1).unsqueeze(1)
    next_q = target_model(next_obs).gather(1, next_actions).squeeze()
    target = rewards + GAMMA * next_q * (1 - dones)

    loss = loss_fn(q, target.detach())
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item()

def episode_loop(env, epsilon):
    obs, _ = env.reset()
    done = False
    total = 0
    episode_losses = []

    while not done:
        action = policy(obs, env, epsilon)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        store(obs, action, reward, next_obs, done)
        obs = next_obs
        total += reward
        if len(buffer) > WARMUP_STEPS:
            loss = train()
            if loss is not None:
                episode_losses.append(loss)

    mean_loss = np.mean(episode_losses) if episode_losses else 0
    return total, mean_loss

def main():
    set_seed(SEED)
    epsilon = EPS_START
    env = gym.make(config["env"]["name"])
    env.reset(seed=SEED)
    env.action_space.seed(SEED)
    rewards_history = []
    epsilon_history = []
    loss_history = []

    for episode in range(NUM_EPISODES):
        r, mean_loss = episode_loop(env, epsilon)
        rewards_history.append(r)
        epsilon_history.append(epsilon)
        loss_history.append(mean_loss)
        epsilon = max(EPS_MIN, epsilon * EPS_DECAY)

        if episode % TARGET_UPDATE_EVERY == 0:
            target_model.load_state_dict(model.state_dict())
            print(episode)

    env.close()
    torch.save(model.state_dict(), "models/model_double_dqn.pth")
    plot_rewards(rewards_history)
    plot_epsilon(epsilon_history)
    plot_loss(loss_history)
    save_csv(rewards_history, epsilon_history, loss_history)


if __name__ == "__main__":
    main()
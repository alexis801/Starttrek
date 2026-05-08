import gymnasium as gym
import torch
import torch.nn as nn
import random
from collections import deque
import torch.optim as optim

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

optimizer = optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.SmoothL1Loss()


epsilon_min = 0.05
epsilon_decay = 0.998
gamma = 0.99
buffer = deque(maxlen=50000)

def store(obs, action, reward, next_obs, done):
    buffer.append((obs, action, reward, next_obs, done))

def policy(obs, env, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()
    obs_t = torch.tensor(obs, dtype=torch.float32)
    return torch.argmax(model(obs_t)).item()

def train():
    batch_size = 64

    if len(buffer) < batch_size:
        return
    batch = random.sample(buffer, batch_size)
    obs, actions, rewards, next_obs, dones = zip(*batch)

    obs = torch.tensor(obs, dtype=torch.float32)
    actions = torch.tensor(actions)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_obs = torch.tensor(next_obs, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)

    q = model(obs).gather(1, actions.unsqueeze(1)).squeeze()
    next_q = target_model(next_obs).max(1)[0]
    target = rewards + gamma * next_q * (1 - dones)

    loss = loss_fn(q, target.detach())
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

def episode_loop(env, epsilon):
    obs, _ = env.reset()
    done = False
    total = 0

    while not done:
        action = policy(obs, env, epsilon)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        store(obs, action, reward, next_obs, done)
        obs = next_obs
        total += reward
        if len(buffer) > 5000:
            train()

    return total

def main():
    epsilon = 1.0
    env = gym.make("LunarLander-v3")
    rewards = []

    for episode in range(2000):
        r = episode_loop(env, epsilon)
        rewards.append(r)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if episode % 10 == 0:
            target_model.load_state_dict(model.state_dict())
            print(episode)

    env.close()
    torch.save(model.state_dict(), "model.pth")

if __name__ == "__main__":
    main()
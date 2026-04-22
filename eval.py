import gymnasium as gym
import torch

model.load_state_dict(torch.load("model.pth"))
model.eval()
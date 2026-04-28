##
## EPITECH PROJECT, 2026
## star-trek
## File description:
## seed.py
##

# fixe les seed aleatoire

import random
import numpy as np
import torch

def set_seed(seed):

    #random en python
    random.seed(seed)

    #numpy use for gymnasium et calcults matrices
    np.random.seed(seed)

    # torch cpu init poids , morceau  du replay buffer
    torch.manual_seed(seed)

    # torch gpu, cuda ok = seed gpu 
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

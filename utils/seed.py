# fixe les seed aleatoire

import random
import numpy as np
import torch

def set_seed(seed):

    #random en python
    random.seed(seed)

    #numpy use for gymnasium et calcults matrices
    np.random.seed(seed)

import copy
import os
import random
import numbers

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

try:
    import ipywidgets
    from tqdm.notebook import trange
except ImportError:
    from tqdm import trange

from kaggle_environments import make

CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

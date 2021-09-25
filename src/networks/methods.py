

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
import math

from src.utils.model_init import *
from src.networks.resunet import SLBR


# our method
def slbr(**kwargs):
    return SLBR(args=kwargs['args'], shared_depth=1, blocks=3, long_skip=True)




# basic python
import sys, os, glob, time, pdb, json, pickle
import numpy as np
#import pandas as pd
from PIL import Image
from tqdm import tqdm
import scipy.io
import importlib
from pathlib import Path

# plotting
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
# import torchist

# utils
from modules.utils.time_remaining import *
from modules.utils.triangle import *

# publication quality plots
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('png', 'pdf') # plt.savefig('name.pdf', format='pdf')

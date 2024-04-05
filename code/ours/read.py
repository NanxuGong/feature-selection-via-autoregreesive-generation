import argparse
import os
import sys

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append('/home/dwangyang/NIPS2023/IJCAI-AutoFS/code')
import pickle
import random
import sys
from typing import List


import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils
from torch import Tensor
from torch.utils.data import DataLoader

from controller import GAFS
from feature_env import FeatureEvaluator, base_path
from utils_meter import AvgrageMeter, pairwise_accuracy, hamming_distance, count_parameters_in_MB, FSDataset
from record import SelectionRecord
from utils.logger import info, error
import time

a = pd.read_hdf(f'{base_path}/history/uci_credit_card.hdf', key='raw_train')
print(a)
a.to_csv('filename.csv', index=False)
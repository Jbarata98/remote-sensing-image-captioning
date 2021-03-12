from configs.enums_file import *
import time
import torch.optim
import torch.utils.data
import os
import json
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample
import cv2

#GLOBAL PARAMETERS
ARCHITECTURE = ARCHITECTURES.BASELINE.value
DATASET = DATASETS.RSICD.value

#TRAINING PARAMETERS
ENCODER_MODEL = EncoderModels.EFFICIENT_NET_IMAGENET_FINETUNE.value
ATTENTION = ATTENTION.soft_attention.value  # todo hard_attention
OPTIMIZER = OPTIMIZERS.ADAM.value
LOSS = LOSSES.Cross_Entropy.value

#PATHS
RSICD_PATH = 'images/RSICD_images'
UCM_PATH = 'images/UCM_images'
SYDNEY_PATH = 'images/SYDNEY_images'
RSICD_CAPTIONS_PATH = 'captions/dataset_rsicd_modified.json'
UCM_CAPTIONS_PATH = 'captions/dataset_ucm_modified.json'
SYDNEY_CAPTIONS_PATH = 'captions/dataset_sydney_modified.json'




from configs.enums_file import *
import torch.backends.cudnn as cudnn
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

cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

#fine tune
FINE_TUNE = True

#GLOBAL PARAMETERS
ARCHITECTURE = ARCHITECTURES.BASELINE.value
DATASET = DATASETS.RSICD.value

#TRAINING PARAMETERS
ENCODER_MODEL = ENCODERS.EFFICIENT_NET_IMAGENET_FINETUNED.value #which encoder using now
ATTENTION = ATTENTION.soft_attention.value  # todo hard_attention
OPTIMIZER = OPTIMIZERS.ADAM.value
LOSS = LOSSES.Cross_Entropy.value

#PATHS
RSICD_PATH = 'images/RSICD_images'
UCM_PATH = 'images/UCM_images'
SYDNEY_PATH = 'images/SYDNEY_images'
#CAPTIONS PATH
RSICD_CAPTIONS_PATH = 'captions/dataset_rsicd_modified.json'
UCM_CAPTIONS_PATH = 'captions/dataset_ucm_modified.json'
SYDNEY_CAPTIONS_PATH = 'captions/dataset_sydney_modified.json'
#INPUT CLASSES PATH
RSICD_CLASSES_PATH = '../classification/classes_rsicd'
UCM_CLASSES_PATH = '../classification/classes_ucm'
SYDNEY_CLASSES_PATH = '../classification/classes_sydney'
#CLASSIFICATION DATASET PATH
RSICD_CLASSIFICATION_DATASET_PATH = "../classification/datasets/classification_dataset_rsicd.json"
UCM_CLASSIFICATION_DATASET_PATH = "../classification/datasets/classification_dataset_ucm.json"
SYDNEY_CLASSIFICATION_DATASET_PATH = "../classification/datasets/classification_dataset_sydney.json"

#FOR EVALUATION
JSON_refs_coco = 'test_coco_format'
bleurt_checkpoint = "metrics_files/bleurt/test_checkpoint"  # uses Tiny

#LOADERS
ENCODER_LOADER = ENCODERS.EFFICIENT_NET_IMAGENET.value #which encoder loading from
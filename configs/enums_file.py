from enum import Enum
import torch
import torchvision
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DATASETS(Enum):
    RSICD = 'rsicd'
    UCM = 'ucm'
    SYDNEY = 'sydney'

class EncoderModels(Enum):
    RESNET = 'resnet'
    EFFICIENT_NET = 'efficient_net'

class ARCHITECTURES(Enum):
    BASELINE = 'SAT_baseline'
    FUSION = 'initial_architecture'

class ATTENTION(Enum):
    soft_attention = 'soft_attention'
    hard_attention = 'hard_attention'
    bottom_up_top_down = 'bottom_up_top_down'

class OPTIMIZER(Enum):
    ADAM = 'Adam'
    Adam_W = 'AdamW'
    Ada_Belief = 'AdaBelief' #todo

class LOSSES(Enum):
    Cross_Entropy = 'CrossEntropy'


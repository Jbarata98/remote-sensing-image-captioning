from enum import Enum
import torch
import torchvision
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DATASETS():
    RSICD = 'rsicd'
    UCM = 'ucm'
    SYDNEY = 'sydney'

class EncoderModels():
    RESNET = torchvision.models.resnet101(pretrained=True)
    EFFICIENT_NET = 'efficient_net'

class ARCHITECTURES():
    BASELINE = 'SAT_baseline'
    FUSION = 'initial_architecture'

class ATTENTION():
    soft_attention = 'soft_attention'
    hard_attention = 'hard_attention'
    bottom_up_top_down = 'bottom_up_top_down'

class OPTIMIZER():
    ADAM = torch.optim.Adam
    Adam_W = torch.optim.AdamW
    Ada_Belief = 'AdaBelief' #todo

class LOSSES():
    Cross_Entropy = nn.CrossEntropyLoss().to(device)


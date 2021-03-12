from enum import Enum


class DATASETS(Enum):
    RSICD = 'rsicd'
    UCM = 'ucm'
    SYDNEY = 'sydney'

class EncoderModels(Enum):
    RESNET = 'resnet'
    EFFICIENT_NET_IMAGENET = 'efficient_net_imagenet'
    EFFICIENT_NET_IMAGENET_FINETUNE = 'efficient_net_imagenet_finetune'

class ARCHITECTURES(Enum):
    BASELINE = 'baseline'
    FUSION = 'initial_architecture'

class ATTENTION(Enum):
    soft_attention = 'soft_attention'
    hard_attention = 'hard_attention'
    bottom_up_top_down = 'bottom_up_top_down'

class OPTIMIZERS(Enum):
    ADAM = 'Adam'
    Adam_W = 'AdamW'
    Ada_Belief = 'AdaBelief' #todo

class LOSSES(Enum):
    Cross_Entropy = 'CrossEntropy'


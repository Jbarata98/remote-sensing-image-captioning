from enum import Enum

class DATASETS(Enum):
    RSICD = 'rsicd'
    UCM = 'ucm'
    SYDNEY = 'sydney'

class ENCODERS(Enum):
    RESNET = 'resnet' #initial tests
    EFFICIENT_NET_IMAGENET = 'efficient_net_imagenet'
    EFFICIENT_NET_IMAGENET_FINETUNED = 'efficient_net_imagenet_finetune'

class DECODERS(Enum):
    FUSION_BASE = 'pegasus_lstm'
    FUSION_TOPDOWN = 'pegasus_lstm_topdown'
    LSTM = 'lstm'

class ARCHITECTURES(Enum):
    BASELINE = 'baseline'
    FUSION = 'fusion'

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


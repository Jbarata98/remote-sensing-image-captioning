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
    FUSION_PEGASUS = 'pegasus_lstm'
    FUSION_GPT2 = 'gpt2_lstm'
    LSTM = 'lstm'
    PEGASUS = 'pegasus'
    GPT2 = 'gpt2'

class ARCHITECTURES(Enum):
    BASELINE = 'baseline'
    FUSION = 'fusion'

class ATTENTION(Enum):
    soft_attention = 'soft_attention'
    hard_attention = 'hard_attention'
    bottom_up_top_down = 'bottomup_topdown'

class OPTIMIZERS(Enum):
    ADAM = 'Adam'
    Adam_W = 'AdamW'
    Ada_Belief = 'AdaBelief' #todo

class LOSSES(Enum):
    Cross_Entropy = 'CrossEntropy'


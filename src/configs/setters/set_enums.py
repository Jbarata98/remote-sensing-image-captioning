from enum import Enum

class DATASETS(Enum):
    RSICD = 'rsicd'
    UCM = 'ucm'
    SYDNEY = 'sydney'

class ENCODERS(Enum):
    """---------------------------------------------RESNET-----------------------------------------------------------"""
    RESNET = 'resnet' #initial tests
    """_____________________________________________EFF_NET__________________________________________________________"""
    EFFICIENT_NET_IMAGENET = 'efficient_net_imagenet'
    EFFICIENT_NET_IMAGENET_FINETUNED = 'efficient_net_imagenet_finetune'
    EFFICIENT_NET_IMAGENET_FINETUNED_AUGMENTED = 'efficient_net_imagenet_finetune_augmented'
    EFFICIENT_NET_IMAGENET_FINETUNED_AUGMENTED_CONTRASTIVE = 'efficient_net_imagenet_finetune_augmented_contrastive'
    """__________________________________________EFF_NET_V2__________________________________________________________"""
    EFFICIENT_NET_V2_IMAGENET = 'efficient_net_imagenet_V2'
    EFFICIENT_NET_V2_IMAGENET_FINETUNED_AUGMENTED = 'efficient_net_V2_imagenet_finetune_augmented_contrastive'
    EFFICIENT_NET_V2_IMAGENET_FINETUNED_AUGMENTED_CONTRASTIVE = 'efficient_net_V2_imagenet_finetune_augmented_contrastive'
    EFFICIENT_NET_V2_IMAGENET_FINETUNED_AUGMENTED_CONTRASTIVE_CE = 'efficient_net_V2_imagenet_finetune_augmented_contrastive_CE'
    EFFICIENT_NET_V2_IMAGENET_FINETUNED_AUGMENTED_CONTRASTIVE_ALS = 'efficient_net_V2_imagenet_finetune_augmented_contrastive_ALS'

class AUX_LMs(Enum):
    PEGASUS = 'pegasus'
    GPT2 = 'gpt2'

class ARCHITECTURES(Enum):
    BASELINE = 'baseline'
    FUSION = 'fusion'

class ATTENTION_TYPE(Enum):
    soft_attention = 'soft_attention'
    hard_attention = 'hard_attention'
    top_down = 'topdown'
    pyramid_attention = 'pyramid_dual'

class OPTIMIZERS(Enum):
    ADAM = 'Adam'
    Adam_W = 'AdamW'
    Ada_Belief = 'AdaBelief' #todo

class LOSSES(Enum):
    Cross_Entropy = 'CrossEntropy'
    SupConLoss = 'SupConLoss'
    ALS = 'ALS'


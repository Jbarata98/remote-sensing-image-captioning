from configs.enums_file import *
from configs.training_details import *
#PARAMETERS
ARCHITECTURE = ARCHITECTURES.BASELINE.value
DATASET = DATASETS.RSICD.value
MODEL = EncoderModels.EFFICIENT_NET_IMAGENET_FINETUNE.value
ATTENTION = ATTENTION.soft_attention.value  # todo hard_attention

#PATHS
RSICD_PATH = 'images/RSICD_images'
UCM_PATH = 'images/UCM_images'
SYDNEY_PATH = 'images/SYDNEY_images'
RSICD_CAPTIONS_PATH = 'captions/dataset_rsicd_modified.json'
UCM_CAPTIONS_PATH = 'captions/dataset_ucm_modified.json'
SYDNEY_CAPTIONS_PATH = 'captions/dataset_sydney_modified.json'




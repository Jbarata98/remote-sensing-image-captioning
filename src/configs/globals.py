from src.configs.enums_file import *
import torch.backends.cudnn as cudnn
import torch.utils.data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# fine tune
FINE_TUNE = False

#custom vocab
CUSTOM_VOCAB = True #True if using transformers vocab and want to create a custom one in order to reduce the size.

# GLOBAL PARAMETERS
ARCHITECTURE = ARCHITECTURES.FUSION.value
DATASET = DATASETS.RSICD.value

# tokenization parameters for AUXLM
SPECIAL_TOKENS = {"bos_token": "<start>",
                  "eos_token": "<end>",
                  "unk_token": "<unk>",
                  "pad_token": "<pad>"}

# TRAINING PARAMETERS
ENCODER_MODEL = ENCODERS.EFFICIENT_NET_IMAGENET.value  # which encoder using now
AUX_LM =  AUX_LMs.GPT2.value  #lstm which decoder using

ATTENTION = ATTENTION.soft_attention.value  # todo hard_attention

OPTIMIZER = OPTIMIZERS.ADAM.value
LOSS = LOSSES.Cross_Entropy.value

# PATHS
RSICD_PATH = '../data/images/RSICD_images'
UCM_PATH = '../data/images/UCM_images'
SYDNEY_PATH = '..data/images/SYDNEY_images'

# CAPTIONS PATH
RSICD_CAPTIONS_PATH = '../data/captions/dataset_rsicd_modified.json'
UCM_CAPTIONS_PATH = '../data/captions/dataset_ucm_modified.json'
SYDNEY_CAPTIONS_PATH = '../data/captions/dataset_sydney_modified.json'

# INPUT CLASSES PATH
RSICD_CLASSES_PATH = '../../data/classification/classes_rsicd'
UCM_CLASSES_PATH = '../data/classification/classes_ucm'
SYDNEY_CLASSES_PATH = '../data/classification/classes_sydney'

# CLASSIFICATION DATASET PATH
RSICD_CLASSIFICATION_DATASET_PATH = "../../data/classification/datasets/classification_dataset_rsicd.json"
UCM_CLASSIFICATION_DATASET_PATH = "../data/classification/datasets/classification_dataset_ucm.json"
SYDNEY_CLASSIFICATION_DATASET_PATH = "../data/classification/datasets/classification_dataset_sydney.json"

# FOR EVALUATION
JSON_refs_coco = 'test_coco_format'
bleurt_checkpoint = "metrics_files/bleurt/test_checkpoint"  # uses Tiny

# LOADERS
ENCODER_LOADER = ENCODERS.EFFICIENT_NET_IMAGENET.value  # which pre-trained encoder loading from
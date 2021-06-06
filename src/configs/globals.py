import torch.utils.data
import torch.backends.cudnn as cudnn
from src.configs.setters.set_enums import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# task {Retrieval,Classification,Captioning}
TASK = 'Captioning'

# if using COLAB
COLAB = False

# fine tune is True if want to unfreeze encoder and decoder
FINE_TUNE = False

# custom vocab

# tokenization parameters for AUXLM
SPECIAL_TOKENS = {"bos_token": "<start>",
                  "eos_token": "<end>",
                  "unk_token": "<unk>",
                  "pad_token": "<pad>"}


# GLOBAL PARAMETERS
ARCHITECTURE = ARCHITECTURES.BASELINE.value
DATASET = DATASETS.RSICD.value

CUSTOM_VOCAB = True   # True if creating a custom vocab in order to reduce the size.

# MODELS
ENCODER_MODEL = ENCODERS.EFFICIENT_NET_IMAGENET.value  # which encoder using now

AUX_LM = AUX_LMs.PEGASUS.value if ARCHITECTURE == ARCHITECTURES.FUSION.value else None  # which aux. LM using

# TRAINING PARAMETERS
ATTENTION = ATTENTION_TYPE.soft_attention.value  # type of attention

OPTIMIZER = OPTIMIZERS.ADAM.value
LOSS = LOSSES.Cross_Entropy.value if TASK == 'Captioning' else LOSSES.SupConLoss.value

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
# which pre-trained encoder loading from/loading to
# if doing classification pretraining  the loader path might be different from the current encoder (pretraining an efficientnet on imagenet)
ENCODER_LOADER = ENCODERS.EFFICIENT_NET_IMAGENET_FINETUNED_AUGMENTED_CONTRASTIVE.value if TASK == 'Classification' else ENCODER_MODEL

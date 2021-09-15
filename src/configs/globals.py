import torch.utils.data
import torch.backends.cudnn as cudnn
from src.configs.setters.set_enums import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# task {Retrieval,Classification,Captioning,Summarization}
"""----------------------------------------------- TASK -------------------------------------------------------------"""
TASK = 'Retrieval'
# if using COLAB
COLAB = False

"""--------------------------------------------- FINE TUNED ---------------------------------------------------------"""
# fine tune is True change paths
FINE_TUNED_PATH = False

"""-------------------------------------------- SPECIAL TOKENS ------------------------------------------------------"""
# tokenization parameters for AUXLM
SPECIAL_TOKENS = {"bos_token": "<start>",
                  "eos_token": "<end>",
                  "unk_token": "<unk>",
                  "pad_token": "<pad>"}

"""-------------------------------------------- GLOBAL PARAMS -------------------------------------------------------"""
# GLOBAL PARAMETERS
ARCHITECTURE = ARCHITECTURES.FUSION.value
DATASET = DATASETS.RSICD.value

CUSTOM_VOCAB = True  # True if creating a custom vocab in order to reduce the size.

"""------------------------------------------------ MODELS ----------------------------------------------------------"""
# MODELS
ENCODER_MODEL = ENCODERS.EFFICIENT_NET_V2_IMAGENET_FINETUNED_AUGMENTED_CONTRASTIVE.value  # which encoder using now

AUX_LM = AUX_LMs.PEGASUS.value if ARCHITECTURE == ARCHITECTURES.FUSION.value else None  # which aux. LM using

"""------------------------------------------- TRAINING PARAMETERS --------------------------------------------------"""
ATTENTION = ATTENTION_TYPE.pyramid_attention.value  # type of attention

OPTIMIZER = OPTIMIZERS.ADAM.value
LOSS = LOSSES.SupConLoss.value if TASK == 'Classification' else LOSSES.Cross_Entropy.value

"""----------------------------------------------- ABLATIONS --------------------------------------------------------"""
if AUX_LM == AUX_LMs.PEGASUS.value:
    # if doing multi_input for pegasus encoder else False
    MULTI_INPUT = False
    REDUCTION_LAYER = True
    """--------Types of Fusion--------"""
    CONCAT_ONLY = False
    SIMPLE_FUSION = True
    #todo
    COLD_FUSION = False
    HIERARCHICAL_FUSION = False

if TASK == 'Classification':
    EXTRA_EPOCHS = True

"""------------------------------------------------- PATHS ----------------------------------------------------------"""
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

"""------------------------------------------------- LOADER ---------------------------------------------------------"""
# LOADERS
# which pre-trained encoder loading from/loading to
# if doing classification pretraining  the loader path might be different from the current encoder (pretraining an efficientnet on imagenet)
ENCODER_LOADER = ENCODERS.EFFICIENT_NET_V2_IMAGENET_FINETUNED_AUGMENTED_CONTRASTIVE.value if TASK == 'Classification' else ENCODER_MODEL
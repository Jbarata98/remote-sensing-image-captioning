from src.configs.getters.get_models import *
from src.configs.getters.get_data_paths import *
from src.configs.getters.get_training_details import *
from src.configs.getters.get_training_optimizers import *
from src.configs.utils.embeddings import *

# Initializers

# set hyperparameters

if torch.cuda.is_available():  # running on colab
    if COLAB:
        HPARAMETER = Training_details("/content/gdrive/MyDrive/Tese/code/src/configs/setters/training_details.txt")

    #virtual GPU
    else:
        pass
else:
    #running locally
    HPARAMETER = Training_details(
        "/src/configs/setters/training_details.txt")

h_parameter = HPARAMETER._get_training_details()

# parameters for main filename
caps_per_img = int(h_parameter['captions_per_image'])
min_word_freq = int(h_parameter['min_word_freq'])

# base name shared by data files {nr of captions per img and min word freq}
data_name = DATASET + "_" + str(caps_per_img) + "_cap_per_img_" + str(
    min_word_freq) + "_min_word_freq"

figure_name = DATASET + "_" + ENCODER_MODEL + "_" + ATTENTION  # when running visualization

# set paths
PATHS = Paths(architecture=ARCHITECTURE, attention=ATTENTION, encoder=ENCODER_MODEL, AuxLM=AUX_LM, filename=data_name,
              figure_name=figure_name, dataset=DATASET, fine_tune=FINE_TUNE)
# set encoder
ENCODER = Encoders(model=ENCODER_MODEL, checkpoint_path=PATHS._load_encoder_path(encoder_name=ENCODER_LOADER),
                   device=DEVICE)
# set AuxLM
AuxLM = AuxLM(model=AUX_LM, device=DEVICE) if ARCHITECTURE == ARCHITECTURES.FUSION.value else None

#if in fact using Aux-LM, load it with special tokens
if AuxLM:
    AuxLM_tokenizer, AuxLM_model = AuxLM._get_decoder_model(special_tokens=SPECIAL_TOKENS)

# set optimizers
OPTIMIZERS = Optimizers(optimizer_type=OPTIMIZER, loss_func=LOSS)

# folder with input data files
data_folder = PATHS._get_input_path()

checkpoint_model = PATHS._get_checkpoint_path()

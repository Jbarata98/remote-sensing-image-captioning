import torch.cuda

from src.configs.get_models import *
from src.configs.get_data_paths import *
from src.configs.get_training_details import *
from src.configs.get_training_optimizers import *
from src.configs.embeddings import *


# Initializers

# set hyperparameters

if torch.cuda.is_available(): #not running locally
    HPARAMETER = Training_details("/content/gdrive/MyDrive/Tese/code/configs/training_details.txt")
else:
    HPARAMETER = Training_details("/src/configs/training_details.txt")

h_parameter = HPARAMETER._get_training_details()

# parameters for main filename
caps_per_img = int(h_parameter['captions_per_image'])
min_word_freq = int(h_parameter['min_word_freq'])

# base name shared by data files {nr of captions per img and min word freq in create_input_files.py}
data_name = DATASET + "_" + str(caps_per_img) + "_cap_per_img_" + str(
    min_word_freq) + "_min_word_freq"  # DATASET + '_CLASSIFICATION_dataset'
figure_name = DATASET + "_" + ENCODER_MODEL + "_" + ATTENTION  # when running visualization

# set paths
PATHS = Paths(architecture=ARCHITECTURE, attention=ATTENTION, encoder=ENCODER_MODEL, AuxLM=AUX_LM, filename=data_name,
              figure_name=figure_name, dataset=DATASET, fine_tune=FINE_TUNE)
# set encoder
ENCODER = Encoders(model=ENCODER_MODEL, checkpoint_path=PATHS._load_encoder_path(encoder_loader=ENCODER_LOADER),
                   device=DEVICE)
# set AuxLM
AuxLM = AuxLM(model = AUX_LM,device=DEVICE) if ARCHITECTURE == ARCHITECTURES.FUSION.value else None

if AuxLM:
    AuxLM_tokenizer, AuxLM_model = AuxLM._get_decoder_model(special_tokens=SPECIAL_TOKENS)

# set optimizers
OPTIMIZERS = Optimizers(optimizer_type=OPTIMIZER, loss_func=LOSS)

# folder with data files saved by create_input_files.py
data_folder = PATHS._get_input_path()
# get_path(ARCHITECTURE, model = MODEL, data_name=data_name,checkpoint = True, best_checkpoint = True, fine_tune = fine_tune_encoder) #uncomment for checkpoint
checkpoint_model = PATHS._get_checkpoint_path()

# name of wordmap (FOR LSTM because its manually created)


def save_checkpoint(epoch, val_loss_improved, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                    decoder_optimizer,
                    bleu4, is_best):

    """
    Saves model checkpoint.
    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    :param bleu4: validation BLEU-4 score for this epoch
    """

    if is_best:
        state = {'epoch': epoch,
                 'epochs_since_improvement': epochs_since_improvement,
                 'bleu-4': bleu4,
                 'encoder': encoder,
                 'decoder': decoder,
                 'encoder_optimizer': encoder_optimizer,
                 'decoder_optimizer': decoder_optimizer}

        filename_best_checkpoint = Paths._get_checkpoint_path()
        torch.save(state, filename_best_checkpoint)

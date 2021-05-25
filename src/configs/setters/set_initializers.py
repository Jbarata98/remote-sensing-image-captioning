from src.configs.getters.get_models import *
from src.configs.getters.get_data_paths import *
from src.configs.getters.get_training_details import *
from src.configs.getters.get_training_optimizers import *
from src.configs.utils.embeddings import *

# Initializers
import sys


# print(sys.path)


# set hyperparameters
class Setters:
    """
    class that sets the parameters for the code
    """

    def __init__(self, file='configs/setters/training_details.txt'):
        self.file = file

    def _set_training_parameters(self):

        if torch.cuda.is_available():  # running on colab
            if COLAB:
                HPARAMETER = Training_details(
                    "/content/gdrive/MyDrive/Tese/code/src/configs/setters/training_details.txt")

            # virtual GPU
            else:
                HPARAMETER = Training_details(self.file)
        else:
            # running locally
            HPARAMETER = Training_details(self.file)

        h_parameter = HPARAMETER._get_training_details()

        return h_parameter

    # parameters for main filename
    def _set_captions_parameters(self):
        h_parameter = self._set_training_parameters()
        caps_per_img = int(h_parameter['captions_per_image'])
        min_word_freq = int(h_parameter['min_word_freq'])
        return {"caps_per_img": caps_per_img,
                "min_word_freq": min_word_freq}

    # base name shared by data files {nr of captions per img and min word freq}
    def _set_base_data_name(self):
        data_name = DATASET + "_" + str(self._set_captions_parameters()["caps_per_img"]) + "_cap_per_img_" + str(
            self._set_captions_parameters()["min_word_freq"]) + "_min_word_freq"
        return data_name

    # name for figure
    def _set_figure_name(self):
        figure_name = DATASET + "_" + ENCODER_MODEL + "_" + ATTENTION  # when running visualization
        return figure_name

    # set paths
    def _set_paths(self):
        paths = Paths(architecture=ARCHITECTURE, attention=ATTENTION, encoder=ENCODER_MODEL, AuxLM=AUX_LM,
                      filename=self._set_base_data_name(),
                      figure_name=self._set_figure_name(), dataset=DATASET, fine_tune=FINE_TUNE)
        return paths

    # set encoder
    def _set_encoder(self, pretrained_encoder = ENCODER_LOADER):
        encoder = Encoders(model=ENCODER_MODEL,
                           checkpoint_path=self._set_paths()._get_pretrained_encoder_path(encoder_name=pretrained_encoder),
                           device=DEVICE)
        return encoder

    # set AuxLM
    def _set_aux_lm(self):
        aux_lm = AuxLM(model=AUX_LM,
                       device=DEVICE) if ARCHITECTURE == ARCHITECTURES.FUSION.value and TASK == 'CAPTIONING' else None

        AuxLM_tokenizer, AuxLM_model = aux_lm._get_decoder_model(
            special_tokens=None if AUX_LM == AUX_LMs.PEGASUS.value else SPECIAL_TOKENS)
        return {"tokenizer": AuxLM_tokenizer,
                "model": AuxLM_model}

    # set optimizers
    def _set_optimizer(self):
        optimizers = Optimizers(optimizer_type=OPTIMIZER, loss_func=LOSS)
        return optimizers

    # folder with input data files
    def _set_input_folder(self):
        data_folder = self._set_paths()._get_input_path()
        return data_folder

    # checkpoint path
    def _set_checkpoint_model(self, is_best = True):
        checkpoint_model = self._set_paths()._get_checkpoint_path(is_best = is_best)
        return checkpoint_model

# import sys
# sys.path.insert(0, '/content/gdrive/MyDrive/Tese/code')  # for colabv

from src.configs.setters.set_initializers import *
from src.captioning_scripts.baseline.train_baseline import TrainBaseline
from src.captioning_scripts.fusion.gpt2.train_gpt2 import TrainGPT2
from src.captioning_scripts.fusion.pegasus.train_pegasus import TrainPegasus

if ARCHITECTURE == ARCHITECTURES.BASELINE.value:
    # # initialize the class
    _train = TrainBaseline(language_aux=None, fine_tune_encoder=False)

elif ARCHITECTURE == ARCHITECTURES.FUSION.value:
    if AUX_LM == AUX_LMs.GPT2.value:
        _train = TrainGPT2(language_aux=AUX_LM, fine_tune_encoder=False)
    elif AUX_LM == AUX_LMs.PEGASUS.value:
        _train = TrainPegasus(language_aux=AUX_LM, fine_tune_encoder=False)

# setup the vocab (size and word map)
_train._setup_vocab()
# initiate the models
_train._init_model()
# load checkpoint if exists
_train._load_weights_from_checkpoint(_train.decoder, _train.decoder_optimizer, _train.encoder,
                                     _train.encoder_optimizer, is_current_best=True)
# load dataloaders (train and val)
_train._setup_dataloaders()
# # setup parameters for training
_train._setup_train(_train._train, _train._validate)

from src.configs.setters.set_initializers import *
from src.captioning_scripts.baseline.train_baseline import TrainBaseline

if ARCHITECTURE == ARCHITECTURES.BASELINE.value:
    # # initialize the class
    _train = TrainBaseline(language_aux=None, fine_tune_encoder=False, checkpoint=None)

elif ARCHITECTURE == ARCHITECTURES.FUSION.value:
    if AUX_LM == AUX_LMs.GPT2.value:
        _train = TrainBaseline(language_aux=AUX_LM, fine_tune_encoder=False, checkpoint=None)
    elif AUX_LM == AUX_LMs.PEGASUS.value:
        _train = TrainBaseline(language_aux=AUX_LM, fine_tune_encoder=False, checkpoint=None)

# setup the vocab (size and word map)
_train._setup_vocab()
# initiate the models
_train._init_model()
# load checkpoint if exists
_train._load_weights_from_checkpoint(Setters()._set_paths())
# load dataloaders (train and val)
_train._setup_dataloaders()
# # setup parameters for training
_train._setup_train(_train._train_baseline, _train._validate_baseline)

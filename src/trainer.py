from src.configs.setters.set_initializers import *
from src.train import TrainEndToEnd

# initialize the class
trainer = TrainEndToEnd(language_aux=aux_lm, fine_tune_encoder=False, checkpoint=None)
# setup the vocab (size and word map if its baseline)
trainer._setup_vocab()
# initiate the models
trainer._init_models()
# load checkpoint if exists
trainer._load_weights_from_checkpoint()
# load dataloaders (train and val)
trainer._setup_dataloaders()
# setup parameters for training
trainer._setup_train()

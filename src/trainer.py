# import sys
# sys.path.insert(0, '/content/gdrive/MyDrive/Tese/code')  # for colab
from src.classification_scripts.SupConLoss.train_supcon import FineTuneSupCon
from src.classification_scripts.cross_entropy.train_ce import FineTuneCE
from src.configs.setters.set_initializers import *
from src.captioning_scripts.baseline.train_baseline import TrainBaseline
from src.captioning_scripts.fusion.gpt2.train_gpt2 import TrainGPT2
from src.captioning_scripts.fusion.pegasus.train_pegasus import TrainPegasus
if TASK == 'Captioning':
    if ARCHITECTURE == ARCHITECTURES.BASELINE.value:
        # # initialize the class
        _train = TrainBaseline(language_aux=None, fine_tune_encoder=False)

    elif ARCHITECTURE == ARCHITECTURES.FUSION.value:
        if AUX_LM == AUX_LMs.GPT2.value:
            _train = TrainGPT2(language_aux=AUX_LM, fine_tune_encoder=False)
        elif AUX_LM == AUX_LMs.PEGASUS.value:
            _train = TrainPegasus(language_aux=AUX_LM, pretrain = False, fine_tune_encoder=False, nr_inputs=1)

    # setup the vocab (size and word map)
    _train._setup_vocab()
    # initiate the models
    _train._init_model()
    # load checkpoint if exists might need inputs variable if its pegasus ( multi input )
    _train._load_weights_from_checkpoint(_train.decoder, _train.decoder_optimizer, _train.encoder, _train.encoder_optimizer, is_current_best=True, nr_inputs = _train.nr_inputs if AUX_LM == AUX_LMs.PEGASUS.value else 1)
    # load dataloaders (train and val)
    _train._setup_dataloaders()
    # # setup parameters for training
    _train._setup_train(_train._train, _train._validate)


elif TASK == 'Classification':
    if LOSS == LOSSES.Cross_Entropy.value:
        logging.basicConfig(
            format='%(levelname)s: %(message)s', level=logging.INFO)
        model = FineTuneCE(model_type=ENCODER_MODEL, device=DEVICE, file = 'classification_scripts/encoder_training_details.txt')
        model._setup_train()
        model._setup_transforms()
        model._setup_dataloaders()
        model.train(model.train_loader, model.val_loader)
    elif LOSS == LOSSES.SupConLoss.value:
        logging.basicConfig(
            format='%(levelname)s: %(message)s', level=logging.INFO)
        model = FineTuneSupCon(model_type=ENCODER_MODEL, device=DEVICE, file = 'classification_scripts/encoder_training_details.txt')
        model._setup_train()
        model._setup_transforms()
        model._setup_dataloaders()
        model.train(model.train_loader, model.val_loader)

#import sys
# sys.path.insert(0, '/content/gdrive/MyDrive/Tese/code')  # for colab

from src.classification_scripts.SupConLoss.train_supcon import FineTuneSupCon
from src.classification_scripts.ALS.train_ALSingle import FineTuneALS
from src.classification_scripts.cross_entropy.train_ce import FineTuneCE
from src.configs.setters.set_initializers import *
from src.captioning_scripts.baseline.train_baseline import TrainBaseline
from src.captioning_scripts.fusion.gpt2.train_gpt2 import TrainGPT2
from src.captioning_scripts.fusion.pegasus.train_pegasus import TrainPegasus

if TASK == 'Captioning':
    if ARCHITECTURE == ARCHITECTURES.BASELINE.value:

        # # initialize the class
        _train = TrainBaseline(language_aux=None, pretrain = False,fine_tune_encoder=False, model_version = 'v2')

    elif ARCHITECTURE == ARCHITECTURES.FUSION.value:
        if AUX_LM == AUX_LMs.GPT2.value:
            _train = TrainGPT2(language_aux=AUX_LM, fine_tune_encoder=False, model_version= 'v2')
        elif AUX_LM == AUX_LMs.PEGASUS.value:
            _train = TrainPegasus(language_aux=AUX_LM, pretrain = False, fine_tune_encoder=False, nr_inputs=1, model_version= 'v2')

    # setup the vocab (size and word map)
    _train._setup_vocab()

    # init model
    _train._init_model()

    # load checkpoint if exists might need inputs variable if its pegasus ( multi input )
    _train._load_weights_from_checkpoint(_train.decoder, _train.decoder_optimizer, _train.encoder, _train.encoder_optimizer, is_current_best=True, nr_inputs = _train.nr_inputs if ARCHITECTURES == ARCHITECTURES.FUSION.value
                                                                                                                                                                                   and AUX_LM == AUX_LMs.PEGASUS.value                                                                                                                                                                         else 1)
    # load dataloaders (train and val)
    _train._setup_dataloaders()

    # setup parameters for training
    _train._setup_train(_train._train_critical if SELF_CRITICAL else _train._train, _train._validate)


elif TASK == 'Classification':
    # to run extra epochs with a different loss
    if EXTRA_EPOCHS:
        logging.basicConfig(
            format='%(levelname)s: %(message)s', level=logging.INFO)
        logging.info('PRETRAINING ENCODER WITH EXTRA EPOCHS ON {}...'.format(LOSS))
        if LOSS == LOSSES.SupConLoss.value:
            model = FineTuneSupCon(model_type=ENCODER_MODEL, device=DEVICE, file = 'classification_scripts/encoder_training_details.txt', eff_net_version = 'v2')
        elif LOSS == LOSSES.Cross_Entropy.value:
            model = FineTuneCE(model_type=ENCODER_MODEL, device=DEVICE, file = 'classification_scripts/encoder_training_details.txt', eff_net_version = 'v2')
        elif LOSS == LOSSES.ALS.value:
            model = FineTuneALS(model_type=ENCODER_MODEL, device=DEVICE, file = 'classification_scripts/encoder_training_details.txt', eff_net_version = 'v2')
        model._setup_train()
        model._setup_transforms()
        model._setup_dataloaders()
        model.train(model.train_loader, model.val_loader)

    else:
        if LOSS == LOSSES.Cross_Entropy.value:
            logging.basicConfig(
                format='%(levelname)s: %(message)s', level=logging.INFO)
            logging.info('PRETRAINING ENCODER WITH CROSS-ENTROPY...')
            model = FineTuneCE(model_type=ENCODER_MODEL, device=DEVICE, file = 'classification_scripts/encoder_training_details.txt', eff_net_version = 'v2')
            model._setup_train()
            model._setup_transforms()
            model._setup_dataloaders()
            model.train(model.train_loader, model.val_loader)

        elif LOSS == LOSSES.SupConLoss.value:
            logging.basicConfig(
                format='%(levelname)s: %(message)s', level=logging.INFO)
            logging.info('PRETRAINING ENCODER WITH SUPERVISED CONTRASTIVE LOSS...')
            model = FineTuneSupCon(model_type=ENCODER_MODEL, device=DEVICE, file = 'classification_scripts/encoder_training_details.txt', eff_net_version = 'v2')
            model._setup_train()
            model._setup_transforms()
            model._setup_dataloaders()
            model.train(model.train_loader, model.val_loader)





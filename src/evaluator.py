# import sys
# sys.path.insert(0, '/content/gdrive/MyDrive/Tese/code')  # for colabv

from src.configs.setters.set_initializers import *

from src.captioning_scripts.fusion.pegasus.eval_pegasus import EvalPegasus
from src.captioning_scripts.fusion.pegasus.train_pegasus import TrainPegasus

# if ARCHITECTURE == ARCHITECTURES.BASELINE.value:
#     # # initialize the class
#     _train = TrainBaseline(language_aux=None, fine_tune_encoder=False)

if ARCHITECTURE == ARCHITECTURES.FUSION.value:
    # if AUX_LM == AUX_LMs.GPT2.value:
    #     _train = TrainGPT2(language_aux=AUX_LM, fine_tune_encoder=False)
    if AUX_LM == AUX_LMs.PEGASUS.value:
        _train = TrainPegasus(language_aux=AUX_LM, fine_tune_encoder=False)
        _train._setup_vocab()
        _train._init_model()
        _eval = EvalPegasus(encoder=_train.encoder, decoder=_train.decoder, aux_lm=_train.aux_lm, hashmap=_train.hashmap, word_map=_train.word_map, vocab_size=_train.vocab_size
                            , sim_mapping=_train.sim_mapping, pegasus_input= _train.pegasus_input, device=_train.device,
                            checkpoint=Setters()._set_checkpoint_model(), b_size=3)

        _eval._load_checkpoint()
    # setup vocab for evaluation

    # get special tokens
    _eval._get_special_tokens()

# setup dataloaders, transformations, etc.
_eval._setup_evaluate()

# eval
_eval._evaluate()

# import sys
# sys.path.insert(0, '/content/gdrive/MyDrive/Tese/code')  # for colabv
import json
import os

from src.configs.setters.set_initializers import *

from src.captioning_scripts.fusion.pegasus.eval_pegasus import EvalPegasus
from src.captioning_scripts.fusion.pegasus.train_pegasus import TrainPegasus
from src.captioning_scripts.baseline.eval_simple import EvalBaseline
from src.captioning_scripts.baseline.eval_topdown import EvalBaselineTopDown
from src.captioning_scripts.baseline.train_baseline import TrainBaseline
from src.captioning_scripts.fusion.gpt2.eval_gpt2 import EvalGPT2
from src.captioning_scripts.fusion.gpt2.train_gpt2 import TrainGPT2
from src.classification_scripts.cross_entropy.test_ce import TestCE
from src.classification_scripts.SupConLoss.test_supcon import TestSupCon
from src.compute_scores import create_json, compute_scores

if TASK == 'CAPTIONING':
    LOAD_HYPOTHESIS = False

    # already evaluated if you want to load the hypothesis only from file
    if LOAD_HYPOTHESIS:
        with open('../' + Setters()._set_paths()._get_hypothesis_path( results_array=True), "rb") as hyp_file:
            hypotheses = pickle.load(hyp_file)

    else:
        if ARCHITECTURE == ARCHITECTURES.BASELINE.value:        # # initialize the class
            _train = TrainBaseline(language_aux=None, fine_tune_encoder=False)
            _train._setup_vocab()
            _train._init_model()
            _train._load_weights_from_checkpoint(decoder =_train.decoder, decoder_optimizer= _train.decoder_optimizer,
                                                 encoder=_train.encoder, encoder_optimizer= _train.encoder_optimizer)
            if ATTENTION == ATTENTION_TYPE.soft_attention.value:
                _eval = EvalBaseline(encoder=_train.encoder, decoder=_train.decoder,word_map=_train.word_map, vocab_size=_train.vocab_size
                                 ,device=_train.device, checkpoint=Setters()._set_checkpoint_model(), b_size=5)
            elif ATTENTION == ATTENTION_TYPE.top_down.value:
                _eval = EvalBaselineTopDown(encoder=_train.encoder, decoder=_train.decoder, word_map=_train.word_map,
                                     vocab_size=_train.vocab_size
                                     , device=_train.device, checkpoint=Setters()._set_checkpoint_model(), b_size=5)



        elif ARCHITECTURE == ARCHITECTURES.FUSION.value:
            if AUX_LM == AUX_LMs.GPT2.value:
                _train = TrainGPT2(language_aux=AUX_LM, fine_tune_encoder=False)
                _train._setup_vocab()
                _train._init_model()
                _train._load_weights_from_checkpoint(decoder=_train.decoder, decoder_optimizer=_train.decoder_optimizer,
                                                     encoder=_train.encoder, encoder_optimizer=_train.encoder_optimizer)
                _eval = EvalGPT2(encoder=_train.encoder, decoder=_train.decoder, aux_lm=_train.aux_lm,
                                 hashmap=_train.hashmap, word_map=_train.word_map, vocab_size=_train.vocab_size
                                 , device=_train.device, checkpoint=Setters()._set_checkpoint_model(), b_size=3)

            elif AUX_LM == AUX_LMs.PEGASUS.value:
                _train = TrainPegasus(language_aux=AUX_LM, fine_tune_encoder=False)
                _train._setup_vocab()
                _train._init_model()
                _train._load_weights_from_checkpoint(decoder=_train.decoder, decoder_optimizer=_train.decoder_optimizer,
                                                     encoder=_train.encoder, encoder_optimizer=_train.encoder_optimizer)
                _eval = EvalPegasus(encoder=_train.encoder, decoder=_train.decoder, aux_lm=_train.aux_lm,
                                    hashmap=_train.hashmap, word_map=_train.word_map, vocab_size=_train.vocab_size
                                    , sim_mapping=_train.sim_mapping, pegasus_input=_train.pegasus_input,
                                    device=_train.device, checkpoint=Setters()._set_checkpoint_model(), b_size=3)

                _eval._load_checkpoint()

                TO_EVALUATE = False
            # setup vocab for evaluation

            # get special tokens
            _eval._get_special_tokens()

        # setup dataloaders, transformations, etc.
        _eval._setup_evaluate()

        # eval
        references, hypotheses = _eval._evaluate()

    # create json with hypothesis
    create_json(hypotheses)

    # computes scores and saves report
    compute_scores()

elif TASK == 'Classification':
    if LOSS == LOSSES.Cross_Entropy.value:
        logging.basicConfig(
            format='%(levelname)s: %(message)s', level=logging.INFO)
        tester = TestCE()
        tester._set_loader()
        tester._setup_model()
        tester._load_checkpoint()
        tester._setup_eval()
        pred_dict = tester._compute_acc()
        # epoch_acc_train = compute_acc(train_loader, "TRAIN")

        # predicted["acc_train"] = epoch_acc_train




    elif LOSS == LOSSES.SupConLoss.value:
        logging.basicConfig(
            format='%(levelname)s: %(message)s', level=logging.INFO)

        tester = TestSupCon()
        tester._set_loader()
        tester._setup_model()
        tester._load_checkpoint()

    # output_path = '../../' + Setters(file="classification_scripts/encoder_training_details.txt")._set_paths()._get_results_path()
    #
    # with open(output_path, 'w+') as f:
    #     json.dump(pred_dict, f, indent=2)



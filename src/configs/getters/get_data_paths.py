import logging
from src.configs.globals import *
import io
import pickle


# Current date time in local system


# -----------------------------------------PATHS---------------------------------------------

class CPU_Unpickler(pickle.Unpickler):  # useful when loading from gpu to cpu (from colab to local)
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


class Paths:

    def __init__(self, architecture=None, attention=None, encoder=None, AuxLM=None, filename=None, figure_name=None,
                 dataset='rsicd', fine_tune=False):

        """
         :param architecture: architecture of the model {SAT_baseline/Fusion}
         :param attention: which attention technique the model is using
         :param encoder: which encoder model are you using
         :param AuxLM: which AuxLM model are you using
         :param figure_name: name of the figure (attention visualization)
         :param fine_tune: is it fine tuned (unfreeze everything)
        """

        self.architecture = architecture
        self.attention = attention
        self.encoder = encoder
        self.AuxLM = AuxLM
        self.filename = filename
        self.figure_name = figure_name
        self.dataset = dataset

        self.fine_tune = fine_tune

    def _get_images_path(self):
        """
        returns image dataset path
        """
        if self.dataset == DATASETS.RSICD.value:
            return RSICD_PATH
        elif self.dataset == DATASETS.UCM.value:
            return UCM_PATH
        elif self.dataset == DATASETS.SYDNEY.value:
            return SYDNEY_PATH
        else:
            logging.error("Wrong dataset // SUPPORTED : rsicd, ucm or sydney")

    def _get_classification_dataset_path(self):
        """
        returns classification dataset path
        """
        if self.dataset == DATASETS.RSICD.value:
            return RSICD_CLASSIFICATION_DATASET_PATH
        elif self.dataset == DATASETS.UCM.value:
            return UCM_CLASSIFICATION_DATASET_PATH
        elif self.dataset == DATASETS.SYDNEY.value:
            return SYDNEY_CLASSIFICATION_DATASET_PATH
        else:
            raise Exception("Invalid dataset")

    def _get_captions_path(self):
        """
        return captions path (.json)
        """
        if self.dataset == DATASETS.RSICD.value:
            return RSICD_CAPTIONS_PATH
        elif self.dataset == DATASETS.UCM.value:
            return UCM_CAPTIONS_PATH
        elif self.dataset == DATASETS.SYDNEY.value:
            return SYDNEY_CAPTIONS_PATH
        else:
            logging.error("Wrong dataset // SUPPORTED : rsicd, ucm or sydney")

    def _get_architectures_path(self):
        """
        return path for architecture (baseline or fusion)
        """
        if self.fine_tune:
            path_architecture = self.architecture + '/fine_tuned/'
        else:
            path_architecture = self.architecture + '/simple/'
        return path_architecture

    def _get_input_path(self, is_classification=False):
        """
        return path for input files
        """
        if is_classification:
            path_input = '../../../experiments/encoder/inputs/'
        else:
            path_input = '../experiments/' + self._get_architectures_path() + 'inputs/'
        return path_input

    def _load_encoder_path(self, encoder_name=None):
        """
        get path to load encoder
        """

        path_checkpoint = 'experiments/encoder/encoder_checkpoints/' + encoder_name + '_checkpoint_.pth.tar'
        return path_checkpoint

    def _get_checkpoint_path(self, classification_task=False):

        """
        get path to save checkpoint files
        """

        if classification_task:
            path_checkpoint = 'experiments/encoder/encoder_checkpoints/' + self.encoder + '_checkpoint_' + '.pth.tar'
        # not for classification task
        else:
            # if is fusion
            if ARCHITECTURE == ARCHITECTURES.FUSION.value:

                path_checkpoint = 'experiments/' + self._get_architectures_path() + 'checkpoints/' + '_checkpoint_' + self.encoder + '_' + self.AuxLM + '_' + self.filename + '.pth.tar'

            # baseline
            else:
                path_checkpoint = 'experiments/' + self._get_architectures_path() + 'checkpoints/' + '_checkpoint_' + self.encoder + '_' + self.filename + '.pth.tar'

        return path_checkpoint

    def _get_hypothesis_path(self, date, results_array=False):
        """
        get path for hypothesis file (generated output)
        """
        if ARCHITECTURE == ARCHITECTURES.FUSION.value:
            #save the results in an array as temporary file
            if results_array:
                path_hypothesis = 'experiments/' + self._get_architectures_path() + 'results/hypothesis.pkl'
            else:
                path_hypothesis = 'experiments/' + self._get_architectures_path() + 'results/' + self.encoder + '_' + self.AuxLM + '_' + date + '_hypothesis.json'
        else:  # is baseline
            path_hypothesis = 'experiments/' + self._get_architectures_path() + 'results/' + self.encoder + '_' + 'hypothesis.json'

        return path_hypothesis

    def _get_test_sentences_path(self):
        """
        get path for hypothesis file (generated output)
        """
        path_test = 'experiments/' + self._get_architectures_path() + 'results/' + self.dataset + '_' + JSON_refs_coco + '.json'
        return path_test

    def _get_results_path(self, results_array=False, bleu_4=0):

        """
        get path for results file (rsicd_test_coco_format.json)
        """
        if ARCHITECTURE == ARCHITECTURES.FUSION.value:
            if results_array:
                path_results = 'experiments/' + self._get_architectures_path() + 'results/references.pkl'
            else:
                path_results = 'experiments/' + self._get_architectures_path() + 'results/' + self.encoder + '_' + self.AuxLM + '_' + 'evaluation_results_BLEU4_' + str(
                    bleu_4) + '_' + self.attention + '.json'
        else:
            path_results = 'experiments/' + self._get_architectures_path() + 'results/' + self.encoder + '_' + 'evaluation_results_' + self.attention + '.json'

        return path_results

    def _get_output_folder_path(self, is_classification=False):

        """
        get path for output folder
        """

        if is_classification:
            path_output = '../experiments/encoder' + self._get_architectures_path() + 'results/'

        else:
            path_output = 'experiments/' + self._get_architectures_path() + 'results/'
        return path_output

    def _get_figure_path(self):
        """
        get path for figure alphas visualization folder
        """
        path_figure = 'experiments/' + self._get_architectures_path() + '/results/' + self.encoder + '_' + self.figure_name + '.png'
        return path_figure

    def _get_features_path(self, split):

        """
        get path for features folder
        """

        path_features = '../experiments/encoder/features/' + self.encoder + '_' + self.dataset + '_features_' + split + '.pickle'
        return path_features

    def _get_index_path(self, split='TRAIN'):

        """
        get path for features folder
        """
        path_index = 'experiments/encoder/indexes/index_' + self.dataset + '_train'
        path_dict = 'experiments/encoder/indexes/' + self.dataset + '_index_dict_' + split + '.pickle'

        return {'path_index': path_index, 'path_dict': path_dict}

    def _get_pegasus_tokenizer_path(self, split='TRAIN'):

        """
        get path for summaries folder
        """

        path_tokenized = 'experiments/' + self._get_architectures_path() + '/results/' + self.dataset + '_pegasus_tokenized_' + split + '.pkl'

        return path_tokenized

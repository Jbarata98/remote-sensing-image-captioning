import logging
from datetime import datetime

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
            path_input = '../experiments/encoder/inputs/'
        else:
            if self.architecture == 'fusion':
                if TASK == 'Summarization':
                    path_input = '../experiments/' + self._get_architectures_path() + 'inputs/' + self.AuxLM + '/pretrain/'
                elif TASK == 'Captioning':
                    path_input = '../experiments/' + self._get_architectures_path() + 'inputs/' + self.AuxLM + '/'
            else:
                path_input = '../experiments/' + self._get_architectures_path() + 'inputs/'
        return path_input

    def _get_pretrained_encoder_path(self, encoder_name=None):
        """
        get path to load/save encoder
        """

        path_checkpoint = 'experiments/encoder/encoder_checkpoints/' + encoder_name + '_checkpoint_.pth.tar'
        return path_checkpoint

    def _get_checkpoint_path(self, is_best=True):

        """
        get path to save checkpoint files
        """
        # if is fusion
        if ARCHITECTURE == ARCHITECTURES.FUSION.value:
            if TASK == 'Summarization':
                path_checkpoint = 'experiments/' + self._get_architectures_path() + 'checkpoints/' + self.AuxLM + '/pretrain/'
            elif TASK == 'Captioning':
                ablation = '_multi_input_' if MULTI_INPUT else '_single_input_'
                if is_best:
                    path_checkpoint = 'experiments/' + self._get_architectures_path() + 'checkpoints/' + self.AuxLM + '/' + 'BEST_checkpoint_' + self.encoder + '_' +   self.AuxLM + ablation +  self.attention  + '_' + self.filename + '.pth.tar'
                else:
                    path_checkpoint = 'experiments/' + self._get_architectures_path() + 'checkpoints/' + self.AuxLM + '/' + '_checkpoint_' + self.encoder + '_'  + self.AuxLM + ablation +  self.attention  + '_' + self.filename + '.pth.tar'

        if ARCHITECTURE == ARCHITECTURES.BASELINE.value:  # baseline
            if is_best:
                path_checkpoint = 'experiments/' + self._get_architectures_path() + 'checkpoints/' + 'BEST_checkpoint_' + self.encoder + '_' + self.attention + '_' + self.filename + '.pth.tar'

            else:
                path_checkpoint = 'experiments/' + self._get_architectures_path() + 'checkpoints/' + '_checkpoint_' + self.encoder + '_' + self.attention + '_' + self.filename + '.pth.tar'
        return path_checkpoint

    def _get_hypothesis_path(self, results_array=False):
        """
        get path for hypothesis file (generated output)
        """
        if ARCHITECTURE == ARCHITECTURES.FUSION.value:
            ablation = '_multi_input_' if MULTI_INPUT else '_single_input_'

            # save the results in an array as temporary file
            if results_array:
                path_hypothesis = 'experiments/' + self._get_architectures_path() + 'results/' + self.AuxLM + '/' + self.encoder + '_' + self.attention + '_' + self.filename + 'hypothesis.pkl'
            else:
                date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
                path_hypothesis = 'experiments/' + self._get_architectures_path() + 'results/' + self.AuxLM + '/' + self.encoder  + '_' + self.attention + '_' + self.AuxLM + ablation + date + '_hypothesis.json'
        else:  # is baseline
            # date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
            if results_array:
                path_hypothesis = 'experiments/' + self._get_architectures_path() + 'results/' + self.encoder + '_' + self.attention + '_' + self.filename + '_hypothesis.pkl'
            else:
                date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")

                path_hypothesis = 'experiments/' + self._get_architectures_path() + 'results/' + self.encoder  + '_' + self.attention + '_' + self.filename + '_' + date + '_hypothesis.json'

        return path_hypothesis

    def _get_test_sentences_path(self):
        """
        get path for hypothesis file (generated output)
        """
        path_test = 'experiments/' + self._get_architectures_path() + 'results/' + self.dataset + '_test_coco_format.json'
        return path_test

    def _get_results_path(self, bleu_4=0):

        """
        get path for results file (rsicd_test_coco_format.json)
        """
        if TASK == 'Captioning':
            if ARCHITECTURE == ARCHITECTURES.FUSION.value:

                if AUX_LM == AUX_LMs.PEGASUS.value:
                    ablation = '_multi_input_' if MULTI_INPUT else '_single_input_'

                    path_results = 'experiments/' + self._get_architectures_path() + 'results/' + self.AuxLM + '/' + self.encoder + '_' + ablation + self.AuxLM + '_' + 'evaluation_results_BLEU4_' + str(
                        bleu_4) + '_' + self.attention + '.json'
                else:
                    path_results = 'experiments/' + self._get_architectures_path() + 'results/' + self.AuxLM + '/' + self.encoder + '_' + self.AuxLM + '_' + 'evaluation_results_BLEU4_' + str(
                        bleu_4) + '_' + self.attention + '.json'

            else:
                path_results = 'experiments/' + self._get_architectures_path() + 'results/' + self.encoder + '_'  + 'evaluation_results_BLEU4_' + str(
                    bleu_4) + '_' + self.attention + '.json'
        elif TASK == 'Classification':
            date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
            path_results = 'experiments/encoder/results/' + self.encoder + '_' + 'evaluation_results_' + date + '.json'
        elif TASK == 'Retrieval':
            date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
            path_results = 'experiments/encoder/results/' + self.encoder + '_' + 'retrieval_evaluation_results_' + date + '.json'
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
        path_figure = 'experiments/' + self._get_architectures_path() + 'results/' + self.encoder + '_' + self.attention + '_' + self.figure_name + '.png'
        return path_figure

    def _get_features_path(self, split):

        """
        get path for features folder
        """

        path_features = '../experiments/encoder/features/' + self.encoder + '_' + self.dataset + '_features_' + split + '.pickle'
        return path_features

    def _get_index_path(self):

        """
        get path for features folder
        """
        path_index = 'experiments/encoder/indexes/index_' + self.encoder + '_' + self.dataset + '_train'
        path_dict = 'experiments/encoder/indexes/index_' + self.encoder + '_' + self.dataset + '_dict_' + '_train' + '.pickle'

        return {'path_index': path_index, 'path_dict': path_dict}

    def _get_pegasus_tokenizer_path(self, split='TRAIN'):

        """
        get path for summaries folder
        """

        path_tokenized = 'experiments/' + self._get_architectures_path() + 'inputs/pegasus/' + self.dataset + '_pegasus_tokenized_' + split + '.pkl'

        return path_tokenized

    def _get_similarity_mapping_path(self, nr_similarities = 1):

        """
        get path for similarty mapping folder
        """
        if nr_similarities > 1:
            path_mapping = 'experiments/' + self._get_architectures_path() + 'inputs/pegasus/' + self.dataset + '_' + self.encoder + '_multi_similarity_mapping' + '.json'
        else:
            path_mapping = 'experiments/' + self._get_architectures_path() + 'inputs/pegasus/' + self.dataset + '_' + self.encoder + '_similarity_mapping' + '.json'


        return path_mapping

    def _get_labelled_images_path(self):

        """
        get path for labelled images folder
        """

        path_labelled_images = 'data/classification/datasets/labelled_images.json'

        return path_labelled_images

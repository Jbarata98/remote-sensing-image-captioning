import logging
import sys
from configs.globals import *


# -----------------------------------------PATHS---------------------------------------------

class Paths:

    def __init__(self, architecture = None, attention=None, model=None, filename=None, figure_name=None,
                 dataset='rsicd', fine_tune=False):

        """
         :param architecture: architecture of the model {SAT_baseline/Fusion}
         :param attention: which attention technique the model is using
         :param model: which encoder model are you using
         :param data_name: name of the directory for the data
         :param figure_name: name of the figure
         :param classification: is it for classification?
         :param input: Boolean is it input?
         :param checkpoint: is it a checkpoint?
         :param checkpoint: is it the best checkpoint?
         :param hypothesis: is it generated hypothesis?
         :param results: results file?
         :param output: evaluation output metrics?
         :param figure: attention visualization with figure?
         :param is_encoder: only fine tuning encoder?
         :param fine_tune: is it fine tuned?
        """

        self.architecture = architecture
        self.attention = attention
        self.model = model
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

    def _get_input_path(self, is_classification = False):
        """
        return path for input files
        """
        if is_classification:
            path_input = '../experiments/encoder/inputs/'
        else:
            path_input = '/experiments/' + self._get_architectures_path() + 'inputs/'
        return path_input

    def _get_checkpoint_path(self, encoder_loader = None, is_loading = False, is_encoder = False):
        """
        get path for checkpoint files
        """
        if is_encoder:
            if is_loading: #if loading we want the previous model name #use only when loading the encoder in get_models
                path_checkpoint = '../experiments/encoder/encoder_checkpoints/' + encoder_loader + '_checkpoint_' + '.pth.tar'
            else: # if saving we want the current model name
                path_checkpoint = '../experiments/encoder/encoder_checkpoints/' + self.model + '_checkpoint_' + '.pth.tar'

        else:
            path_checkpoint = '/experiments/' + self._get_architectures_path() + 'checkpoints/' + '_checkpoint_' + self.model + '_' + self.filename + '.pth.tar'
        return path_checkpoint

    def _get_hypothesis_path(self):
        """
        get path for hypothesis file (generated output)
        """
        path_hypothesis = '/experiments/' + self._get_architectures_path() + 'results/' + self.model + '_' + 'hypothesis.json'
        return path_hypothesis

    def _get_results_path(self):
        """
        get path for results file (test_coco_format.json)
        """
        path_results = '/experiments/' + self._get_architectures_path() + 'results/' + self.model + '_' + 'evaluation_results_' + self.attention + '.json'
        return path_results

    def _get_output_folder_path(self, is_classification = False):
        """
        get path for output folder
        """
        if is_classification:
            path_output = '../experiments/encoder' + self._get_architectures_path() + 'results/'

        else:
            path_output = '/experiments/' + self._get_architectures_path() + 'results/'
        return path_output

    def _get_figure_path(self):
        """
        get path for figures folder
        """
        path_figure = '/experiments/' + self._get_architectures_path() + '/results/' + self.model + '_' + self.figure_name + '.png'
        return path_figure



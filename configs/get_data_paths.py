import logging
from configs.globals import *
from configs.training_details import fine_tune_encoder

#-----------------------------------------PATHS---------------------------------------------

def get_images_path(dataset_name):
    if dataset_name == DATASETS.RSICD.value:
        return RSICD_PATH
    elif dataset_name == DATASETS.UCM.value:
        return UCM_PATH
    elif dataset_name == DATASETS.SYDNEY.value:
        return SYDNEY_PATH
    else:
        logging.error("Wrong dataset // SUPPORTED : rsicd, ucm or sydney")

def get_classification_dataset_path(dataset_name):
    if DATASET == DATASETS.RSICD.value:
        return RSICD_CLASSIFICATION_DATASET_PATH
    elif DATASET == DATASETS.UCM.value:
        return UCM_CLASSIFICATION_DATASET_PATH
    elif DATASET == DATASETS.SYDNEY.value:
        return SYDNEY_CLASSIFICATION_DATASET_PATH
    else:
        raise Exception("Invalid dataset")

def get_captions_path(dataset_name):
    if dataset_name == DATASETS.RSICD.value:
        return RSICD_CAPTIONS_PATH
    elif dataset_name == DATASETS.UCM.value:
        return UCM_CAPTIONS_PATH
    elif dataset_name == DATASETS.SYDNEY.value:
        return SYDNEY_CAPTIONS_PATH
    else:
        logging.error("Wrong dataset // SUPPORTED : rsicd, ucm or sydney")

def get_architectures_path(architecture, fine_tune = True):
    if fine_tune:
        path_architecture = architecture + '/fine_tuned/'
    else:
        path_architecture = architecture + '/simple/'
    return path_architecture

#returns data path for chosen variables
def get_path(architecture=None, attention = None,model = None,data_name = None,figure_name = None,
                classification = False, input = False, checkpoint = False, best_checkpoint = False,
                hypothesis = False,results = False, output = False, figure = False, is_encoder = False,
                fine_tune=True):
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
    if input:
        if classification:
            PATH = '../classification/inputs/'
        else:
            PATH = get_architectures_path(architecture,fine_tune) + 'inputs/'

    elif checkpoint:
        if best_checkpoint:
            PATH =  get_architectures_path(architecture,fine_tune) + 'checkpoints/' +  'BEST_checkpoint_' + model + '_' + data_name + '.pth.tar'
        else:
            PATH = get_architectures_path(architecture,fine_tune)  + 'checkpoints/' + '_checkpoint_' + model + '_' + data_name + '.pth.tar'

    elif hypothesis:
        PATH = get_architectures_path(architecture,fine_tune) + 'results/' + model + '_' + 'hypothesis.json'

    elif results:
        PATH = get_architectures_path(architecture,fine_tune) + 'results/' + model + '_' + 'evaluation_results_' + attention + '.json'

    elif output:
        PATH = get_architectures_path(architecture,fine_tune) + 'results/'

    elif figure:
        PATH = get_architectures_path(architecture,fine_tune) + '/results/'  + model + '_' + figure_name + '.png'

    elif is_encoder:
        if best_checkpoint:
            PATH = 'encoder_scripts/encoder_checkpoints/' + model + 'BEST_checkpoint_' + '.pth.tar'
        else:
            PATH = 'encoder_scripts/encoder_checkpoints/' + model + '_checkpoint_' + '.pth.tar'

    else:
        print("Wrong Parameters")
    return PATH

#EVALUATIONS files
data_folder = get_path(ARCHITECTURE, classification= True, input=True, fine_tune = fine_tune_encoder) # folder with data files saved by create_input_files.py
data_name = DATASET + '_CLASSIFICATION_dataset' #DATASET +"_5_cap_per_img_2_min_word_freq"     # base name shared by data files {nr of captions per img and min word freq in create_input_files.py}

checkpoint_model = None #get_path(ARCHITECTURE, model = MODEL, data_name=data_name,checkpoint = True, best_checkpoint = True, fine_tune = fine_tune_encoder) #uncomment for checkpoint
word_map_file = data_folder + 'WORDMAP_' + data_name + '.json'  # word map, ensure it's the same the data was encoded with and the model was trained with

#RESULTS file
JSON_refs_coco = 'test_coco_format'
bleurt_checkpoint = "bleurt/test_checkpoint"  # uses Tiny

JSON_generated_sentences = get_path(architecture=ARCHITECTURE, model=ENCODER_MODEL, hypothesis=True, fine_tune=fine_tune_encoder)
JSON_test_sentences =  get_path(architecture=ARCHITECTURE, model=ENCODER_MODEL,output=True, fine_tune=fine_tune_encoder) +  JSON_refs_coco +'.json'

#evaluation_results = get_path(architecture=ARCHITECTURE, attention = ATTENTION, model=ENCODER_MODEL, results=True, fine_tune=fine_tune_encoder)
#out_file = open(evaluation_results, "w")
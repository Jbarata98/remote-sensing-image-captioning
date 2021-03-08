from configs.datasets import *
from configs.utils import *
from configs.enums_file import *
from configs.paths_generator import *
from configs.training_details import *

ARCHITECTURE = ARCHITECTURES.BASELINE.value


data_folder = PATH_DATA(ARCHITECTURE, input=True, fine_tune = False) # folder with data files saved by create_input_files.py
data_name = DATASETS.RSICD.value +'_5_cap_per_img_2_min_word_freq'  # base name shared by data files {nr of captions per img and min word freq in create_input_files.py}
checkpoint = None #PATH_DATA(ARCHITECTURE, data_name = data_name, checkpoint = True, best_checkpoint = True, fine_tune = False)
word_map_file = data_folder + 'WORDMAP_' + data_name + '.json'  # word map, ensure it's the same the data was encoded with and the model was trained with

#evaluation
ATTENTION = ATTENTION.soft_attention.value  # todo hard_attention
JSON_refs_coco = 'test_coco_format'
bleurt_checkpoint = "bleurt/test_checkpoint"  # uses Tiny

JSON_generated_sentences = PATH_DATA(architecture=ARCHITECTURE, hypothesis=True, fine_tune=False)
JSON_test_sentences =  PATH_DATA(architecture=ARCHITECTURE, output=True, fine_tune=False) +  JSON_refs_coco +'.json'

evaluation_results = PATH_DATA(architecture=ARCHITECTURE, attention = ATTENTION, results=True, fine_tune=False)
out_file = open(evaluation_results, "w")


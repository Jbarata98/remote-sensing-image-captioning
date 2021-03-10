from configs.datasets import *
from configs.enums_file import *
from configs.paths_generator import *
from configs.training_details import *
from configs.getters import *

ARCHITECTURE = ARCHITECTURES.BASELINE.value
DATASET = DATASETS.RSICD.value
MODEL = EncoderModels.EFFICIENT_NET_IMAGENET_FINETUNE.value
ATTENTION = ATTENTION.soft_attention.value  # todo hard_attention

#evaluation files
data_folder = PATH_DATA(ARCHITECTURE, input=True, fine_tune = True) # folder with data files saved by create_input_files.py
data_name = DATASET +'_5_cap_per_img_2_min_word_freq'  # base name shared by data files {nr of captions per img and min word freq in create_input_files.py}
checkpoint =None #PATH_DATA(ARCHITECTURE, model = MODEL, data_name=data_name,checkpoint = True, best_checkpoint = True, fine_tune = False) #uncomment for checkpoint
word_map_file = data_folder + 'WORDMAP_' + data_name + '.json'  # word map, ensure it's the same the data was encoded with and the model was trained with

#results file
JSON_refs_coco = 'test_coco_format'
bleurt_checkpoint = "bleurt/test_checkpoint"  # uses Tiny

JSON_generated_sentences = PATH_DATA(architecture=ARCHITECTURE, model=MODEL, hypothesis=True, fine_tune=True)
JSON_test_sentences =  PATH_DATA(architecture=ARCHITECTURE, model=MODEL,output=True, fine_tune=True) +  JSON_refs_coco +'.json'

evaluation_results = PATH_DATA(architecture=ARCHITECTURE, attention = ATTENTION, model=MODEL, results=True, fine_tune=True)
out_file = open(evaluation_results, "w")


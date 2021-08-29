# script to verify if current test file (rsicd_test_coco_format) has the same captions

import json
import os
import sys
from collections import defaultdict

from src.configs.getters.get_data_paths import *

# script that creates a json with the image names and corresponding labels
PATHS = Paths()

# hard-coded
data_folder = '/home/starksultana/Documentos/MEIC/5o_ano/Tese/code/remote-sensing-image-captioning/data/paths'

def get_image_name(dataset = 'remote_sensing'):
    # get captions path to retrieve image name
    train_filenames = []
    test_dict = defaultdict(list)
    if dataset == 'remote_sensing':
        file = open('../' + PATHS._get_captions_path())
        data = json.load(file)
        for image in data['images']:
            if image['split'] == "test":
                for sentence in image['sentences']:
                    test_dict[image['filename']].append(sentence['raw'])


# print(get_image_name())

def check_phrases_diff(file_1,file_2):
    file_1 = open(file_1)
    file_2 = open(file_2)
    ref_1 = json.load(file_1)
    ref_2 = json.load(file_2)
    counter = 0
    for image_id_1,image_id_2 in zip(ref_1,ref_2):
        if image_id_1['caption'].lstrip() != image_id_2['caption']:
            counter+=1
            print('----------', image_id_1['image_id'])
            print('27 june:',image_id_1['caption'].lstrip())
            print('29 aug:',image_id_2['caption'])
    print("number of different captions:", counter)


# check_phrases_diff('/home/starksultana/Documentos/MEIC/5o_ano/Tese/code/remote-sensing-image-captioning/experiments/fusion/simple/results/pegasus/efficient_net_imagenet_finetune_augmented_contrastive_pegasus_single_input_2021_06_27-05:59:51_PM_hypothesis.json','/home/starksultana/Documentos/MEIC/5o_ano/Tese/code/remote-sensing-image-captioning/experiments/fusion/simple/results/pegasus/efficient_net_imagenet_finetune_augmented_contrastive_soft_attention_pegasus_single_input_2021_08_29-02:29:18_PM_hypothesis.json')
# script to verify if current test file (rsicd_test_coco_format) has the same captions

import json
import os
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


print(get_image_name())


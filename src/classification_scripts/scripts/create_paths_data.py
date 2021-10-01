import json
import os

from src.configs.getters.get_data_paths import *

# script that creates a json with the image names and corresponding labels
PATHS = Paths()

# hard-coded
data_folder = '/home/starksultana/Documentos/MEIC/5o_ano/Tese/code/remote-sensing-image-captioning/data/paths'

def get_image_name(root_path,split, dataset = 'remote_sensing'):
    # get captions path to retrieve image name
    train_filenames = []

    if dataset == 'remote_sensing':
        file = open('../../' + PATHS._get_captions_path())
        data = json.load(file)
        for image in data['images']:
            if image['split'] == split.lower():
                train_filenames.append(root_path + '/' + image['filename'])

    # using another dataset by chance (flickr,coco,etc)
    else:
        file = root_path
        for root, dirs, files in os.walk(file):
            for filename in files:
                train_filenames.append(file + '/' + filename)

    with open(os.path.join(data_folder, split + '_IMGPATHS_' + DATASET + '.json'), 'w') as f:
        json.dump(train_filenames, f)

splits = ["TRAIN", "VAL", "TEST"]
for split in splits:
    get_image_name(root_path="/data/images/UCM_images", split = split)


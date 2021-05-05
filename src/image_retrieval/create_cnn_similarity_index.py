import json
import os

import numpy as np
from tqdm import tqdm

from src.configs.getters.get_data_paths import *
import faiss


PATHS = Paths()

features_list = feature_list = pickle.load(open('../' + PATHS._get_features_path('TRAIN'), 'rb'))


def flatten_maps(feature_list):
    f_maps = []
    # flatten the feature maps
    for fmaps in feature_list:
        for fmap in fmaps:

            fmap = fmap.flatten(start_dim=0, end_dim=1)  # (1,7,7,2048) feature map
            fmap = fmap.mean(dim=0)

            f_maps.append(fmap)  # (2048) dimension feature map

    return f_maps


def get_image_name(dataset = 'remote_sensing'):
    # get captions path to retrieve image name
    train_filenames = []

    if dataset == 'remote_sensing':
        file = open(PATHS._get_classification_dataset_path())
        data = json.load(file)
        for image in data['images']:
            if image['split'] == "train":
                train_filenames.append([image['filename'], image['label']])

    #using another dataset by chance (flickr,coco,etc)
    else:
        file = "/home/starksultana/Documentos/MEIC/5o_ano/Tese/code/remote-sensing-image-captioning/data/images/flickr8k"
        for root, dirs, files in os.walk(file):
            for filename in files:
                train_filenames.append(file + '/' + filename)
    return train_filenames


def create_index(feature_list):
    feature_maps = flatten_maps(feature_list)

    dimensions = feature_maps[0].shape[0]  # 2048

    image_files = get_image_name(dataset = 'remote_sensing')

    index = faiss.IndexFlatL2(dimensions)

    index_dict = {}

    for id, (feature, image_name) in enumerate(zip(tqdm(feature_maps), image_files)):

        image_dict = {id: image_name}

        index_dict.update(image_dict)

        # ids_list = np.linspace(id, id, num=feature.shape[0], dtype="int64")

        # index.
        # index.train(np.array(feature.unsqueeze(0)))
        index.add(np.array(feature.unsqueeze(0)))
    print(index_dict)
    return index, index_dict


if __name__ == '__main__':
    index, index_dict = create_index(feature_list)
    print("Writing index...")
    faiss.write_index(index, '../../' + PATHS._get_index_path()['path_index'])
    print("Writing dict...")
    with open( '../../'+ PATHS._get_index_path()['path_dict'], 'wb+') as f:
        pickle.dump(index_dict, f, True)

    f.close()

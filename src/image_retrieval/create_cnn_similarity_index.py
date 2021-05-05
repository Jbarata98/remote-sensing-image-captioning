import json
import os

import numpy as np
from tqdm import tqdm
from src.image_retrieval.aux_functions import flatten_maps, get_image_name
from src.configs.getters.get_data_paths import *
import faiss

PATHS = Paths()

features_list = feature_list = pickle.load(open('../' + PATHS._get_features_path('TRAIN'), 'rb'))

def create_index(feature_list):
    """
    creates the index given the features extracted
    """

    # flatten the feature maps representation to [2048]
    feature_maps = flatten_maps(feature_list)

    dimensions = feature_maps[0].shape[0]  # 2048

    # get the file names to map them afterwards
    image_files = get_image_name(PATHS, dataset='remote_sensing')

    faiss_index = faiss.IndexFlatL2(dimensions)

    mapping_dict = {}

    for id, (feature, image_name) in enumerate(zip(tqdm(feature_maps), image_files)):
        image_dict = {id: image_name}

        mapping_dict.update(image_dict)

        faiss_index.add(np.array(feature.unsqueeze(0)))

    return index, index_dict


if __name__ == '__main__':
    index, index_dict = create_index(feature_list)
    print("Writing index...")
    faiss.write_index(index, '../../' + PATHS._get_index_path()['path_index'])
    print("Writing dict...")
    with open('../../' + PATHS._get_index_path()['path_dict'], 'wb+') as f:
        pickle.dump(index_dict, f, True)

    f.close()

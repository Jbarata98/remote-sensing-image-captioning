import numpy as np
from tqdm import tqdm
from src.image_retrieval.aux_functions import flatten_maps, get_image_name
from src.configs.getters.get_data_paths import *

from src.image_retrieval.cnn_feature_extractor import PATHS,batch_size
import faiss



# {file_name: feature}
features_dict = feature_list = pickle.load(open('../' + PATHS._get_features_path('train'), 'rb'))

def create_index(features):
    """
    creates the index given the features extracted
    """

    # flatten the feature maps representation to [2048]
    feature_maps = flatten_maps(features, batch_size = batch_size)

    # hard-coded 2048 into dimensions
    dimensions = 2048

    faiss_index = faiss.IndexFlatL2(dimensions)

    mapping_dict = {}
    for id, (image_name, feature) in enumerate(tqdm(feature_maps.items())):
        image_dict = {id: image_name}

        mapping_dict.update(image_dict)

        faiss_index.add(np.array(feature.unsqueeze(0)))
    return faiss_index, mapping_dict


if __name__ == '__main__':
    index, index_dict = create_index(features_dict)
    print("Writing index...")
    faiss.write_index(index, '../../' + PATHS._get_index_path()['path_index'])
    print("Writing dict...")
    with open('../../' + PATHS._get_index_path()['path_dict'], 'wb+') as f:
        pickle.dump(index_dict, f, True)

    f.close()

from encoder_scripts.create_similarity_index import features_list, get_image_name, PATHS

import faiss
import pickle
import numpy as np

class search_index():
    """
    class that receives a single image and outputs the most similar ( or 2nd if its training)
    """

    def __init__(self, feature_map, mode='TRAIN'):

        self.feature_map = feature_map
        self.mode = mode

    def _get_image(self):

        self.index = faiss.read_index(PATHS._get_index_path()['index'])

        self.index_dict = pickle.load(PATHS._get_index_path()['index_dict'])

        self.fmap_flat = np.ascontiguousarray(self.feature_map.flatten(start_dim=0, end_dim=2))

        self.scores, self.neighbors = self.index.search(self.fmap_flat, k=20)


        values,counts = np.unique(self.neighbors.flatten(),return_counts = True)
        print(np.argsort(counts, axis=0)[-2])
        print(get_image_name()[0])

        print(get_image_name()[169])
        print(values[np.argsort(np.max(counts, axis=0))[-2]])
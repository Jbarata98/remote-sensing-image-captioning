
from encoder_scripts.create_similarity_index import get_image_name, PATHS
from PIL import Image

import faiss
import pickle
import numpy as np

#
features_list = feature_list = pickle.load(open(PATHS._get_features_path('TRAIN'), 'rb'))

class search_index():

    """
    class that receives a single image and outputs the most similar ( or 2nd if its training)
    """

    def __init__(self, feature_map, mode='TRAIN'):

        self.feature_map = feature_map
        self.mode = mode


    def _get_image(self, display=False):

        self.display = display

        self.index = faiss.read_index(PATHS._get_index_path()['path_index'])

        with open(PATHS._get_index_path()['path_dict'], "rb") as dict_file:

            self.index_dict = pickle.load(dict_file)

        self.fmap_flat = np.ascontiguousarray(self.feature_map.flatten(start_dim=0, end_dim=2))

        self.scores, self.neighbors = self.index.search(self.fmap_flat, k=15)

        self.values,self.counts = np.unique(self.neighbors.flatten(),return_counts = True)

        if self.mode == 'TRAIN':

            self.id = np.argsort(self.counts, axis=0)[-2]
            img_name = self.index_dict[self.id]
            if display:
                print("Displaying current image...")
                img = Image.open("../" + PATHS._get_images_path() + "/" + self.index_dict[np.argsort(self.counts,axis=0)[-1]])
                img.show()
                print("Displaying target image...")
                img = Image.open("../" +  PATHS._get_images_path() + "/" +  img_name)
                img.show()

            return img_name

        else:

            self.id = np.argsort(self.counts, axis=0)[-1]

            img_name = self.index_dict[self.id]

            if display:

                print("Displaying target image...")
                img = Image.open( "../" + PATHS._get_images_path() + "/" + img_name)
                img.show()

            return img_name


search = search_index(feature_list[0])
search._get_image(display=True)
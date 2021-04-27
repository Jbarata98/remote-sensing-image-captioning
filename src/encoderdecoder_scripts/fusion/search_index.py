
from src.encoderdecoder_scripts.fusion.create_similarity_index import PATHS
from PIL import Image

import faiss
import pickle
import numpy as np
import collections

#todo REFACTOR **NOT WORKING PROPERLY**
features_list = feature_list = pickle.load(open('../../' +PATHS._get_features_path('TRAIN'), 'rb'))

class search_index():

    """
    class that receives a single image and outputs the most similar ( or 2nd if its training)
    """

    def __init__(self, feature_map, mode='TRAIN'):

        self.feature_map = feature_map
        self.mode = mode


    def _get_image(self, display=False):

        self.display = display

        self.index = faiss.read_index('../../../' + PATHS._get_index_path()['path_index'])

        with open('../../../' + PATHS._get_index_path()['path_dict'], "rb") as dict_file:

            self.index_dict = pickle.load(dict_file)

        self.fmap_flat = self.feature_map.flatten(start_dim=0, end_dim=2).mean(dim=0)

        self.scores, self.neighbors = self.index.search(np.array(self.fmap_flat.unsqueeze(0)), k=2)
        # results_dict = collections.defaultdict(int)
        # for (region, neighbors) in zip(self.scores, self.neighbors):
        #     for score, id in zip(region, neighbors):
        #         results_dict[id] += score

        # sorted_dict = dict(sorted(results_dict.items(), key=lambda item: item[1]))


        self.values,self.counts = np.unique(self.neighbors.flatten(),return_counts = True)
        print(self.scores)
        print(self.values,self.counts)
        if self.mode == 'TRAIN':

            # self.arg_nr = np.argsort(self.counts, axis=0)[-2]

            self.id = self.values[1]

            # self.id = list(sorted_dict)[-5]

            img_name = self.index_dict[self.id]
            print(img_name)
            if display:
                print("Displaying current image...")
                # self.arg_nr = np.argsort(self.counts, axis=0)[-1]
                #
                self.id = self.values[0]
                # self.id = list(sorted_dict)[-1]
                img = Image.open("../../" + PATHS._get_images_path() + "/" + self.index_dict[self.id])
                img.show()
                print("Displaying target image...")
                img = Image.open("../../" +  PATHS._get_images_path() + "/" +  img_name)
                img.show()

            return img_name

        else:

            self.id = np.argsort(self.counts, axis=0)[-1]

            img_name = self.index_dict[self.id]

            if display:

                print("Displaying target image...")
                img = Image.open( "../../" + PATHS._get_images_path() + "/" + img_name)
                img.show()

            return img_name

    def _get_captions(self): #todo
        pass


search = search_index(feature_list[15])
search._get_image(display=True)




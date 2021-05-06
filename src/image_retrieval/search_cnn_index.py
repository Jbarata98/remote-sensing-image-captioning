from matplotlib import pyplot as plt

from src.configs.setters.initializers import PATHS

import faiss
import pickle
import numpy as np

#todo REFACTOR **NOT WORKING PROPERLY**

class SearchIndex:

    """
    - receives image and display just for testing purposes
    - write similar image to dict for further mapping when pegasus tokenizing
    """

    def __init__(self, feature_map, split='TRAIN', k = 2):

        self.feature_map = feature_map
        self.split = split
        self.k = k

    def _get_image(self, display=False):

        self.display = display

        # load index
        self.index = faiss.read_index('../../' + PATHS._get_index_path()['path_index'])

        # dictionary to map ids to images
        with open('../../' + PATHS._get_index_path()['path_dict'], "rb") as dict_file:
            self.index_dict = pickle.load(dict_file)

        # flatten
        self.fmap_flat = self.feature_map.flatten(start_dim=0, end_dim=1).mean(dim=0)

        # actual search
        self.scores, self.neighbors = self.index.search(np.array(self.fmap_flat.unsqueeze(0)), k = 8734)

        # results_dict = collections.defaultdict(int)
        # for (region, neighbors) in zip(self.scores, self.neighbors):
        #     for score, id in zip(region, neighbors):
        #         results_dict[id] += score

        # sorted_dict = dict(sorted(results_dict.items(), key=lambda item: item[1]))


        self.values,self.counts = np.unique(self.neighbors.flatten(),return_counts = True)
        print(self.neighbors)
        print(self.values,self.counts)


        # self.arg_nr = np.argsort(self.counts, axis=0)[-2]

        self.id = self.values[0]

        # self.id = list(sorted_dict)[-5]

        self.relevant_neighbors = []
        for file_index in self.neighbors[0]:
            if self.index_dict[file_index][1] == self.index_dict[self.neighbors[0][0]][1]:
                self.relevant_neighbors.append(file_index)

        if display:

            # self.arg_nr = np.argsort(self.counts, axis=0)[-1]
            #
            # print("Displaying target image...")
            #
            # img = Image.open("../../" +  PATHS._get_images_path() + "/" +  target_img)
            # # img = Image.open(target_img)
            # img.show()
            # print("target_img:", target_img)
            #
            # print("Displaying pred image...")
            # self.id = self.values[1]
            # pred_img = self.index_dict[self.id]
            # print("predicted_img:", pred_img)
            # # self.id = list(sorted_dict)[-1]
            # img = Image.open("../../" + PATHS._get_images_path() + "/" + pred_img)
            # # img = Image.open(pred_img)
            fig, ax = plt.subplots(3, 3, figsize=(15, 15))
            #save only the ones from same class


            print("class of reference image:", self.index_dict[self.neighbors[0][0]][1])
            for file_index, ax_i in zip(self.relevant_neighbors, np.array(ax).flatten()):
                # print(file_index)
                #if the class of the target image is the same as the referece image, return
                # print(self.index_dict[file_index][1] )

                # print("found same class")
                ax_i.imshow(plt.imread(
                    "../" + PATHS._get_images_path() + "/" + self.index_dict[file_index][0]))

            plt.show()
        # return pred_img

        else:

            # if split is train return 2nd most similar
            if self.split == 'TRAIN':
                ref_img = #TODO
                target_img = self.index[self.relevant_neighbors[1]][0]
            # if split is validation or test return most similar
            else:
                ref_img = #TODO
                target_img = self.index[self.relevant_neighbors[0]][0]
            return ref_img, target_img

    def _get_captions(self): #todo
        pass

splits = {'train','val','test'}
features_list = pickle.load(open('../' +PATHS._get_features_path('TRAIN'), 'rb'))
search = SearchIndex(,split = 'TRAIN')
search._get_image(display=False)




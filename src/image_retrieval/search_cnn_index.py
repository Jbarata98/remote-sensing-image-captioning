import json

from matplotlib import pyplot as plt
from tqdm import tqdm

from src.configs.setters.set_initializers import *

import faiss
import pickle
import numpy as np

PATHS = Setters('../configs/setters/training_details.txt')._set_paths()


class SearchIndex:
    """
    - receives image and display just for testing purposes
    - write similar image path to dict for further mapping when tokenizing with pegasus (pre-compute)
    """

    def __init__(self, ref_img, feature_map, faiss_index=None, index_dict=None, region_search=False, intra_class=False, split='TRAIN',k=2):

        self.ref_img = ref_img
        self.feature_map = feature_map
        self.split = split
        self.region_search = region_search
        self.intra_class = intra_class
        self.index = faiss_index
        self.index_dict = index_dict
        self.k = k

    def _get_image(self, display=False):

        self.display = display

        # flatten
        self.fmap_flat = self.feature_map.flatten(start_dim=0, end_dim=2).mean(dim=0)

        # actual search
        self.scores, self.neighbors = self.index.search(np.array(self.fmap_flat.unsqueeze(0)), k=5)

        # for region searching
        # use only if not flattening maps (avg pool)
        if self.region_search:
            results_dict = collections.defaultdict(int)
            for (region, neighbors) in zip(self.scores, self.neighbors):
                for score, id in zip(region, neighbors):
                    results_dict[id] += score

            sorted_dict = dict(sorted(results_dict.items(), key=lambda item: item[1]))

        self.values, self.counts = np.unique(self.neighbors.flatten(), return_counts=True)

        self.id = self.values[0]

        with open('../../' + PATHS._get_labelled_images_path(), 'r') as labelled_images:
            image_n_label = json.load(labelled_images)

        self.relevant_neighbors = []
        for file_index in self.neighbors[0]:
            # hack to get the images from same class only
            if self.intra_class:
                if image_n_label.get(self.index_dict[file_index])["Label"] == image_n_label.get(self.ref_img)["Label"]:
                    self.relevant_neighbors.append(file_index)
            else:
                self.relevant_neighbors.append(file_index)

        # display map with 6 most similar images according to the index
        if display:

            fig, ax = plt.subplots(3, 3, figsize=(15, 15))

            for file_index, ax_i in zip(self.relevant_neighbors, np.array(ax).flatten()):
                ax_i.imshow(plt.imread("../" + PATHS._get_images_path() + "/" + self.index_dict[file_index]))

            plt.show()

        else:

            # if split is train return 2nd most similar (most similar actually itself)
            if self.split == 'train':
                ref_img = self.ref_img
                target_img = self.index_dict[self.relevant_neighbors[1]]

            # if split is validation or test return most similar
            else:
                ref_img = self.ref_img
                target_img = self.index_dict[self.relevant_neighbors[0]]
            return ref_img, target_img


def test_faiss(feature_split='train', image_name='baseballfield_120.jpg'):
    features_list = pickle.load(open('../' + PATHS._get_features_path(feature_split), 'rb'))
    # print(features_list.keys())

    # load index
    index = faiss.read_index('../../' + PATHS._get_index_path()['path_index'])

    # dictionary to map ids to images
    with open('../../' + PATHS._get_index_path()['path_dict'], "rb") as dict_file:
        id_dic = pickle.load(dict_file)

    search = SearchIndex(ref_img=None, feature_map=features_list[image_name], faiss_index=index, index_dict=id_dic, split=None)
    search._get_image(display=True)


def create_mappings():
    """
    creates dict with most similar images
    """
    # lower-case because the dataset captions has the splits with lower-cased letters
    splits = ['train', 'val', 'test']
    similarity_dict = collections.defaultdict(dict)

    # load index
    index = faiss.read_index('../../' + PATHS._get_index_path()['path_index'])
    # dictionary to map ids to images
    with open('../../' + PATHS._get_index_path()['path_dict'], "rb") as dict_file:
        id_dic = pickle.load(dict_file)

    for split in splits:
        features_list = pickle.load(open('../' + PATHS._get_features_path(split), 'rb'))
        for img_name, feature in tqdm(features_list.items()):
            search = SearchIndex(ref_img=img_name, feature_map=feature, faiss_index=index, index_dict=id_dic,
                                 split=split)
            ref_img, target_img = search._get_image(display=False)
            similarity_dict[ref_img] = {'Most similar(s)': target_img}

    with open('../../' + PATHS._get_similarity_mapping_path(), 'w+') as f:
        json.dump(similarity_dict, f, indent=2)


# run and create the similarity mappings
if __name__ == '__main__':
    logging.info("testing faiss...")
    test_faiss()
    logging.info("creating the mappings...")
    create_mappings()

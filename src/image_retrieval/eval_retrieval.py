import os

from src.image_retrieval.search_cnn_index import SearchIndex, PATHS
import json

class EvalRetrieval:
    """
    class to evaluate the retrieval part of the architecture
    computes f1-measure and intra class accuracy
    """

    def __init__(self, image_labels, labels):
        self.img_labels = image_labels
        self.labels = labels

        print(self.img_labels, self.labels)


with open('../../' + PATHS._get_labelled_images_path(), 'r') as labelled_images:
    image_n_label = json.load(labelled_images)


with open(os.path.join('../' + PATHS._get_input_path(is_classification=True), 'DICT_LABELS_' + '.json'), 'r') as labels:
    labels = json.load(labels)

evaluator = EvalRetrieval(image_labels=image_n_label, labels = labels)
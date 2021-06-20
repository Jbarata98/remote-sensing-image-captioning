import collections
import os

import numpy as np

from src.image_retrieval.search_cnn_index import PATHS
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import json


class EvalRetrieval:
    """
    class to evaluate the retrieval part of the architecture
    computes f1-measure and intra class accuracy
    """

    def __init__(self, sim_mapping, image_labels, labels):
        self.img_labels = image_labels
        self.labels = labels
        self.mapping = sim_mapping
        self.result = collections.defaultdict(dict)

    def _calc_fmeasure(self):
        self.true = []
        self.pred = []
        for key, item in self.mapping.items():
            self.true.append(self.labels.get(self.img_labels.get(key)['Label']))
            self.pred.append(self.labels.get(self.img_labels.get(item['Most similar'])['Label']))

        score = f1_score(self.true, self.pred, average='macro')
        self.result["f-measure"] = score

    def _class_accuracy(self):
        self.acc_dict = collections.defaultdict(int)
        for key, item in self.mapping.items():
            self.acc_dict[self.img_labels.get(key)['Label']] = {'Total': 0, 'Correct': 0}
        for key, item in self.mapping.items():
            self.acc_dict[self.img_labels.get(key)['Label']]['Total'] += 1
            if self.img_labels.get(key)['Label'] == self.img_labels.get(item['Most similar'])['Label']:
                self.acc_dict[self.img_labels.get(key)['Label']]['Correct'] += 1

        self.result["accuracy_classes"] = self.acc_dict

    def _plot_barchat(self):
        labels = list(self.acc_dict.keys())
        total = []
        correct = []
        for key in self.acc_dict:
            total.append(self.acc_dict[key]['Total'])
            correct.append(self.acc_dict[key]['Correct'])

        # print(labels,total,correct)

        x = np.arange(len(labels))  # the label locations
        width = 0.3  # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width / 2, total, width, label='Total')
        rects2 = ax.bar(x + width / 2, correct, width, label='Correct')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Scores')
        ax.set_title('Correct and Total by Class')
        ax.set_xticks(x)
        ax.set_xticklabels(labels,rotation = 60, ha='right')
        ax.legend()


        fig.tight_layout()
        plt.figure(figsize=(10, 5))  # this creates a figure 8 inch wide, 4 inch high
        plt.show()



with open('../../' + PATHS._get_labelled_images_path(), 'r') as labelled_images:
    image_n_label = json.load(labelled_images)

with open(os.path.join('../' + PATHS._get_input_path(is_classification=True), 'DICT_LABELS_' + '.json'), 'r') as labels:
    labels = json.load(labels)

with open('../../' + PATHS._get_similarity_mapping_path(), 'r') as sim_mapping:
    sim_mapping = json.load(sim_mapping)

evaluator = EvalRetrieval(sim_mapping=sim_mapping, image_labels=image_n_label, labels=labels)
# evaluator._calc_fmeasure()
evaluator._class_accuracy()
# evaluator._plot_barchat()

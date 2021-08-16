import collections
import json

from tqdm import tqdm
from os import listdir
from os.path import isfile, join

from src.configs.setters.set_initializers import *
from src.captioning_scripts.fusion.pegasus.pretrain_pegasus import prepare_data
from nltk.translate.bleu_score import corpus_bleu

# if true only calculate metrics
METRICS = True


class EvalPretrain:
    """
    class to eval Pretrained Model on local data
    """

    def __init__(self, pretrained):
        self.setters = Setters(file='../../../configs/setters/training_details.txt')

        self.paths = self.setters.set_paths()


        self.eval_dic = collections.defaultdict(dict)

        self.AuxLM = self.setters._set_aux_lm(pretrain=pretrained)

    def get_data(self):
        """
        fetch the data
        """
        with open('../../../' + self.paths._get_input_path() + 'raw_captions_dataset', 'r') as captions_file:
            captions_dataset = json.load(captions_file)

        with open('../../../' + self.paths._get_input_path() + 'target_captions_dataset', 'r') as target_file:
            target_dataset = json.load(target_file)

        with open('../../../../' + self.paths._get_similarity_mapping_path(nr_similarities=1), 'r') as hashmap_file:
            hashmap = json.load(hashmap_file)

        train_dict, target_train_dict = captions_dataset["train"], target_dataset["train"]
        test_dict, target_test_dict = captions_dataset["test"], target_dataset["test"]

        test_texts = [' '.join(train_dict.get(hashmap.get(img_name)['Most similar'])) for img_name in test_dict.keys()]
        test_labels = [' '.join(target_train_dict.get(hashmap.get(img_name)['Most similar'])) for img_name in
                       target_test_dict.keys()]

        return test_texts, test_labels

    def tokenize_data(self, text):
        """
        tokenizes the data for pegasus input
        """
        encodings = self.AuxLM["tokenizer"](text, truncation=True, padding='longest', return_tensors="pt").to(DEVICE)
        return encodings

    def get_results(self):
        """
        get targets
        """
        out_file = open("../../../../experiments/fusion/fine_tuned/results/pretrain/pegasus_xsum.json", "w")
        texts, labels = self.get_data()
        for count, (item, label) in tqdm(enumerate(zip(texts, labels))):
            # hack to remove labels ( this dataset was for trainer in huggingface)
            text = self.tokenize_data(item)
            summarized = self.AuxLM["model"].generate(**text)
            text = self.AuxLM["tokenizer"].batch_decode(summarized, skip_special_tokens=True)
            self.eval_dic[count] = {'target': text, 'ref': label}

        json.dump(self.eval_dic, out_file)

        out_file.close()


# function to compute metrics
def compute_metrics(result_dic, doc):
    with open('../../../../experiments/fusion/fine_tuned/results/pretrain/' + doc, 'r') as results_file:
        results = json.load(results_file)

    references_list, targets_list = [], []
    for id in results.keys():

        targets_list.append(results[id]['target'][0].split(' '))
        references_list.append([results[id]['ref'].split(' ')])

    assert len(references_list) == len(targets_list)
    result_dic[doc] = corpus_bleu(references_list,targets_list)

    with open('../../../../experiments/fusion/fine_tuned/results/pretrain/bleu_scores/scores', 'w') as bleu_scores:
        json.dump(result_dic,bleu_scores)

if METRICS:
    files = [f for f in listdir('../../../../experiments/fusion/fine_tuned/results/pretrain/') if
             isfile(join('../../../../experiments/fusion/fine_tuned/results/pretrain/', f))]
    bleu_dict = {}
    for file in files:
        compute_metrics(bleu_dict,file)



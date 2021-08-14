import collections
import json

from tqdm import tqdm

from src.configs.setters.set_initializers import *
from src.captioning_scripts.fusion.pegasus.pretrain_pegasus import prepare_data

setters = Setters(file='../../../configs/setters/training_details.txt')

paths = setters._set_paths()

train_dataset, val_dataset, test_dataset, auxLM = prepare_data()

eval_dic = collections.defaultdict(dict)

def get_data():
    with open('../../../' + paths._get_input_path() + 'raw_captions_dataset', 'r') as captions_file:
        captions_dataset = json.load(captions_file)

    with open('../../../' + paths._get_input_path() + 'target_captions_dataset', 'r') as target_file:
        target_dataset = json.load(target_file)

    with open('../../../../' + paths._get_similarity_mapping_path(nr_similarities=1), 'r') as hashmap_file:
        hashmap = json.load(hashmap_file)

    train_dict, target_train_dict = captions_dataset["train"], target_dataset["train"]
    test_dict, target_test_dict = captions_dataset["test"], target_dataset["test"]

    test_texts = [' '.join(train_dict.get(hashmap.get(img_name)['Most similar'])) for img_name in test_dict.keys()]
    test_labels = [' '.join(target_train_dict.get(hashmap.get(img_name)['Most similar'])) for img_name in
                   target_test_dict.keys()]

    AuxLM = setters._set_aux_lm(pretrain = True)

    return test_texts,test_labels,AuxLM

def tokenize_data(text):
    """
    tokenizes the data for pegasus input
    """
    encodings = AuxLM["tokenizer"](text, truncation=True, padding='longest',return_tensors = "pt").to(DEVICE)
    return encodings


texts, labels, AuxLM = get_data()


# text,labels = tokenize_data()
#

out_file = open("../../../../experiments/fusion/fine_tuned/results/pretrain/pegasus_xsum.json", "w")

for count,(item,label) in tqdm(enumerate(zip(texts,labels))):
    #hack to remove labels ( this dataset was for trainer in huggingface)
    text = tokenize_data(item)
    summarized = auxLM["model"].generate(**text)
    text = auxLM["tokenizer"].batch_decode(summarized,skip_special_tokens = True)
    eval_dic[count] = {'target' : text, 'ref' : label}


json.dump(eval_dic, out_file)

out_file.close()



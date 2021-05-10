import json
import os
import itertools

from src.configs.getters.get_models import *

import collections

from src.configs.setters.set_initializers import *

paths = Setters._set_paths()


class PegasusCaptionTokenizer:
    """
    class to tokenize captions from dataset (already in .json format)
    """

    def __init__(self, file_name='../../' + paths._get_captions_path(), model=AuxLM):

        self.file_name = file_name
        self.model = model

    def _tokenize(self):
        """
        extracts captions (list with 5 captions)
        tokenizes the captions
        """
        tokenized_dict = collections.defaultdict(list)
        with open(self.file_name) as captions_file:
            captions = json.load(captions_file)
            for img_id in captions['images']:
                if img_id['split'] == 'train':
                    for sentence in img_id['sentences']:
                        tokenized_dict[img_id['filename']].append(sentence['raw'])

        tokenizer, model = self.model._get_decoder_model()

        for img, captions in tokenized_dict.items():
            tokenized_dict[img] = tokenizer(captions, truncation=True, padding='longest', return_tensors="pt").to(
                DEVICE)

        return tokenized_dict


class CreateInputPegasus:
    """
    class to create the input to initialize the pegasus encoder
    """

    def __init__(self, captions_name, split, hashmap_name, paths_name='../../../' + paths._get_captions_path(),
                 model=AuxLM):
        self.captions_file = captions_name
        self.split = split
        self.hashmap_name = hashmap_name
        self.paths_file = paths_name
        self.model = model

    def _concat_convert(self):
        """
        extracts captions (list with 5 captions)
        captions already tokenized with custom vocab
        convert them to auxLM tokenizer ids
        """
        captions_dict = collections.defaultdict(list)
        with open('../../../../' + self.paths_file, 'r') as paths_file:
            paths_list = json.load(paths_file)
        with open('../../../' + self.captions_file, 'r') as captions_file:
            captions = json.load(captions_file)
        with open('../../../' + self.hashmap_name, 'r') as hashmap_file:
            hashmap = json.load(hashmap_file)

        for i, path in enumerate(paths_list):
            path = path.split("/")[-1]  # get last because corresponds to the name of the image
            i *= 5
            print(i)
            # create the input for pegasus without pads/start and already converted
            special_tokens = [0, 2645, 2644]
            captions_dict[path] = [hashmap.get(str(tok)) for tok in
                                   list(itertools.chain.from_iterable(captions[i:i + 5])) if
                                   tok not in special_tokens] + [Setters._set_aux_lm()["tokenizer"].eos_token_id]

        with open(os.path.join('../../../' + paths._get_input_path(), split + '_PEGASUS_INPUT_' + '.json'), 'w') as fp:
            json.dump(captions_dict, fp)

            #
        # tokenizer, model = self.model._get_decoder_model()
        #
        # for img, captions in tokenized_dict.items():
        #     tokenized_dict[img] = tokenizer(captions, truncation=True, padding='longest', return_tensors="pt").to(
        #         DEVICE)
        #
        # return tokenized_dict


if __name__ == '__main__':
    # run above script(s)

    splits = ['TRAIN', 'VAL', 'TEST']

    base_filename = DATASET + '_' + str(5) + '_cap_per_img_' + str(
        int(Setters._set_captions_parameters(["min_word_freq"]))) + '_min_word_freq'
    # name of the hashmap for conversion
    hashmap_name = os.path.join(paths._get_input_path(), 'PEGASUS_HASHMAP_' + base_filename + '.json')

    for split in splits:
        captions_name = os.path.join(paths._get_input_path(), split + '_CAPTIONS_' + base_filename + '.json')
        paths_name = os.path.join('data/paths', split + '_IMGPATHS_' + DATASET + '.json')

        tokenizer = CreateInputPegasus(captions_name=captions_name, split=split, hashmap_name=hashmap_name,
                                       paths_name=paths_name)

        tokenizer._concat_convert()

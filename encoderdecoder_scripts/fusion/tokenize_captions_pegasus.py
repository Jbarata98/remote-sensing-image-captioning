from configs.globals import *
from configs.get_models import *
from configs.get_data_paths import *
import collections
import torch
import io

AuxLM = AuxLM(model=AUX_LM, device=DEVICE)

PATHS = Paths()


class captions_tokenizer():
    """
    class to tokenize captions from dataset (already in .json format)
    """

    def __init__(self, file_name=PATHS._get_captions_path(), model=AuxLM):

        self.file_name = file_name
        self.model = model

    def _tokenize(self):
        """
        extracts captions (list with 5 captions)
        tokenizes the captions and saves them in dict
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


tokenizer = captions_tokenizer()

print(tokenizer._tokenize())



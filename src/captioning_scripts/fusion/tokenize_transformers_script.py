import json

from src.configs.setters.set_initializers import *
from src.configs.getters.get_models import *
from src.configs.getters.get_data_paths import *

AuxLM = AuxLM(model=AUX_LM, device=DEVICE)

PATHS = Paths()

aux_lm_tokenizer = Setters._set_aux_lm()['tokenizer']

class TransformersTokenizer:
    """
    class to tokenize captions with transformers tokenizer from dataset and add the same to the dataset (already in .json format)
    """

    def __init__(self, file_name= '../../' + PATHS._get_captions_path(), model=AuxLM):
        self.file_name = file_name
        self.model = model

    def _tokenize(self):
        """
        extracts captions (list with 5 captions)
        tokenizes the captions and adds them to dataset
        """

        with open(self.file_name,"r") as captions_file:
            print("creating new dataset with transformer tokens...")
            captions = json.load(captions_file)
            for img_id in captions['images']:
                for sentence in img_id['sentences']:
                    # print()
                    sentence["tokens_transformers"] = aux_lm_tokenizer.convert_ids_to_tokens(aux_lm_tokenizer(sentence["raw"], add_prefix_space = True)["input_ids"])

        with open(self.file_name,"w") as outcaptions_file:
            print("dumping the new modified dataset...")
            json.dump(captions, outcaptions_file)

tokenizer = TransformersTokenizer()

tokenizer._tokenize()

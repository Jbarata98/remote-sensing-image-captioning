from src.configs.initializers import *
from src.configs.get_models import *
from src.configs.get_data_paths import *

AuxLM = AuxLM(model=AUX_LM, device=DEVICE)

PATHS = Paths()


class wordpiece_tokenizer:
    """
    class to tokenize captions from dataset and add the same to the dataset (already in .json format)
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
            print("creating new dataset with wordpiece tokens...")
            captions = json.load(captions_file)
            for img_id in captions['images']:
                for sentence in img_id['sentences']:
                    cap = " " + sentence["raw"]
                    # print(cap)
                    sentence["tokens_wordpiece"] = AuxLM_tokenizer.convert_ids_to_tokens(AuxLM_tokenizer(cap)["input_ids"])

        with open(self.file_name,"w") as outcaptions_file:
            print("dumping the new modified dataset...")
            json.dump(captions, outcaptions_file)

tokenizer = wordpiece_tokenizer()

tokenizer._tokenize()

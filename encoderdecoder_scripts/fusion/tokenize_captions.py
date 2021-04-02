from configs.globals import *
from configs.get_models import *
from configs.get_data_paths import *
import collections
import torch
import io
DECODER = Decoders(model = DECODER_MODEL,device=DEVICE)

PATHS = Paths()


class captions_tokenizer():

    """
    class to tokenize captions from dataset (already in .json format)
    """

    def __init__(self,file_name = PATHS._get_captions_path(), model = DECODER):

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

        tokenizer,model = self.model._get_decoder_model()

        for img,captions in tokenized_dict.items():
            tokenized_dict[img] = tokenizer(captions, truncation=True, padding='longest', return_tensors="pt").to(DEVICE)

        return tokenized_dict


tokenizer = captions_tokenizer()

print(tokenizer._tokenize())






#
## translated = model.generate(**batch)
# tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
# print("Model:", model)
# print('Summary:', tgt_text[0])
#
# # tokenizer = PegasusTokenizer.from_pretrained(model_name)
# # model = PegasusForConditionalGeneration.from_pretrained(model_name)
# #
# # #
# # # create ids of encoded input vectors
# # input_ids = tokenizer(src_text, return_tensors="pt")
# # # create BOS token
# # decoder_input_ids = tokenizer(tokenizer.eos_token, add_special_tokens=False, return_tensors="pt").input_ids
# # assert decoder_input_ids[0, 0].item() == model.config.decoder_start_token_id, "`decoder_input_ids` should correspond to `model.config.decoder_start_token_id`"
# # # pass input_ids to encoder and to decoder and pass BOS token to decoder to retrieve first logit
# # outputs = model(input_ids.input_ids, decoder_input_ids=decoder_input_ids, return_dict=True, output_hidden_states=True)
# # encoded_sequence = (outputs.encoder_last_hidden_state,)
# # # STEP 11 a
# # lm_logits = outputs.logits
# # lm_states = outputs.decoder_hidden_states[-1]
# # # The two lines below show how to use Bart do decode the most likely word for this position
# # # Instead, you should concatenate the vector lm_states[:, -1:] to the LSTM input, and then use the LSTM to decode
# # next_decoder_input_ids = torch.argmax(lm_logits[:, -1:], axis=-1)
# # decoder_input_ids = torch.cat([decoder_input_ids, next_decoder_input_ids], axis=-1)
# # # STEP 2
# # outputs = model(None, encoder_outputs=encoded_sequence, decoder_input_ids=decoder_input_ids, return_dict=True, output_hidden_states=True)
# # lm_logits = outputs.logits
# # lm_states = outputs.decoder_hidden_states[-1]
# # # The two lines below show how to use Bart do decode the most likely word for this position
# # # Instead, you should concatenate the vector lm_states[:, -1:] to the LSTM input, and then use the LSTM to decode
# # next_decoder_input_ids = torch.argmax(lm_logits[:, -1:], axis=-1)
# # decoder_input_ids = torch.cat([decoder_input_ids, next_decoder_input_ids], axis=-1)
# # # STEP 3, STEP 4, ... repeat the same instructions for all steps of decoding
# #
# # x = torch.tensor([[1,2,3],[1,4,5]])
# # print(x.view(1,-1,1))


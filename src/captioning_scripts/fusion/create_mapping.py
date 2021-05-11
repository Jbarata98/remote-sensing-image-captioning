import json
import os

from src.configs.setters.set_initializers import *


word_map_file = os.path.join(Setters()._set_input_folder(), 'WORDMAP_' + Setters._set_base_data_name() + '.json')
with open('../../' + word_map_file, 'r') as j:
    word_map = json.load(j)
    vocab_size = len(word_map)

#reverse the dictionary
word_map = {y: x for x, y in word_map.items()}

#convert for transformer
hashmap = {y: Setters._set_aux_lm()["tokenizer"].convert_tokens_to_ids(x) for y,x in word_map.items()}

if AUX_LM == AUX_LMs.GPT2.value:
    word_map_file = os.path.join(Setters()._set_input_folder(), 'GPT2_HASHMAP_' + Setters()._set_base_data_name() + '.json')
if AUX_LM == AUX_LMs.PEGASUS.value:
    word_map_file = os.path.join(Setters()._set_input_folder(), 'PEGASUS_HASHMAP_' + Setters()._set_base_data_name() + '.json')


print("saving hashmap to {}".format(word_map_file))

with open('../../' + word_map_file, 'w') as j:
    json.dump(hashmap,j)

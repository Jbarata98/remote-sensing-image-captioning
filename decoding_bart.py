from transformers import BartForConditionalGeneration, BartTokenizer
import torch
# tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
# model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
# # create ids of encoded input vectors
# input_ids = tokenizer("Input sentence", return_tensors="pt")
# # create BOS token
# decoder_input_ids = tokenizer(tokenizer.eos_token, add_special_tokens=False, return_tensors="pt").input_ids
# assert decoder_input_ids[0, 0].item() == model.config.decoder_start_token_id, "`decoder_input_ids` should correspond to `model.config.decoder_start_token_id`"
# # pass input_ids to encoder and to decoder and pass BOS token to decoder to retrieve first logit
# outputs = model(input_ids.input_ids, decoder_input_ids=decoder_input_ids, return_dict=True, output_hidden_states=True)
# encoded_sequence = (outputs.encoder_last_hidden_state,)
# # STEP 1
# lm_logits = outputs.logits
# lm_states = outputs.decoder_hidden_states[-1]
# # The two lines below show how to use Bart do decode the most likely word for this position
# # Instead, you should concatenate the vector lm_states[:, -1:] to the LSTM input, and then use the LSTM to decode
# next_decoder_input_ids = torch.argmax(lm_logits[:, -1:], axis=-1)
# decoder_input_ids = torch.cat([decoder_input_ids, next_decoder_input_ids], axis=-1)
# # STEP 2
# outputs = model(None, encoder_outputs=encoded_sequence, decoder_input_ids=decoder_input_ids, return_dict=True, output_hidden_states=True)
# lm_logits = outputs.logits
# lm_states = outputs.decoder_hidden_states[-1]
# # The two lines below show how to use Bart do decode the most likely word for this position
# # Instead, you should concatenate the vector lm_states[:, -1:] to the LSTM input, and then use the LSTM to decode
# next_decoder_input_ids = torch.argmax(lm_logits[:, -1:], axis=-1)
# decoder_input_ids = torch.cat([decoder_input_ids, next_decoder_input_ids], axis=-1)
# # STEP 3, STEP 4, ... repeat the same instructions for all steps of decoding

# x = torch.tensor([[1,2,3],[1,4,5]])
# print(x.view(1,-1,1))


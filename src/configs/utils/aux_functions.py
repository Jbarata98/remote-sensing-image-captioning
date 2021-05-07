from src.configs.setters.set_initializers import *

def save_captions(caption, captions,LM,freq, max_len):
    if LM == AUX_LMs.GPT2.value:
        tokens = 'tokens_wordpiece'
    elif LM == AUX_LMs.PEGASUS.value:
        tokens = 'tokens_transformers'
    else:
        tokens = 'tokens'
    if len(caption[tokens]) <= max_len:
        if CUSTOM_VOCAB:
                # Update word frequency
                freq.update(caption[tokens])
                # if we want a custom vocab
                captions.append(caption[tokens])

        # tokenize and use the entirety of the vocab provided by the tokenizer
        else:
            captions.append(caption['raw'])

    return freq,captions

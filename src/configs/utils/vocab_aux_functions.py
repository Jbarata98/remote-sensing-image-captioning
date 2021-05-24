from src.configs.setters.set_initializers import *


def save_captions(caption, captions, LM, freq, max_len):
    """
    function to save captions according to its tokenization scheme
    """
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

    return freq, captions


def set_wordmap(words):
    """
    create word_map given the words in the dataset
    """
    # words = [w for w in word_freq.keys()]
    if AUX_LM == AUX_LMs.PEGASUS.value:
        # we need a custom_wordmap if dealing only with LSTM or don't want to use the full pegasus vocab to avoid overhead
        word_map = {k: v + 1 for v, k in enumerate(words)}
        word_map['<start>'] = len(word_map) + 1
        word_map['<pad>'] = 0
    elif AUX_LM == AUX_LMs.PEGASUS.value:
        # we need a custom_wordmap if dealing only with LSTM or don't want to use the full gpt2 vocab to avoid overhead
        word_map = {k: v + 1 for v, k in enumerate(words)}
        word_map['<start>'] = len(word_map) + 1
        word_map['<end>'] = len(word_map) + 1
        word_map['<pad>'] = 0
    #
    # using baseline decoder (uses unk)
    else:
        word_map = {k: v + 1 for v, k in enumerate(words)}
        word_map['<unk>'] = len(word_map) + 1
        word_map['<start>'] = len(word_map) + 1
        word_map['<end>'] = len(word_map) + 1
        word_map['<pad>'] = 0

    return word_map


def encode_captions(tokenizer, c, word_map, max_len, enc_captions, caplens):

    if ARCHITECTURE == ARCHITECTURES.BASELINE.value:
        # using baseline vocab
        enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
            word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))

        # Find caption lengths
        c_len = len(c) + 2

        enc_captions.append(enc_c)
        caplens.append(c_len)

        # if its Transformer-based, need to encode differently
    else:
        if not CUSTOM_VOCAB:
            # if not using custom vocab use tokens straight from tokenizer
            enc_c = tokenizer(SPECIAL_TOKENS[
                                        'bos_token'] + c +
                                    SPECIAL_TOKENS['eos_token'], truncation=True, max_length=35,
                                    padding="max_length")

            enc_captions.append(enc_c['input_ids'])

            # not using UNKs with GPT2
            caplens.append(enc_c['attention_mask'].count(1))

        else:
            # Encode captions for custom vocab

            if AUX_LM == AUX_LMs.PEGASUS.value:
                # already add the end token
                enc_c = [word_map['<start>']] + [word_map.get(word) for word in c] + [word_map['<pad>']] * (max_len - len(c))
                c_len = len(c) + 1

            else:
                enc_c = [word_map['<start>']] + [word_map.get(word) for word in c] + [word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))
                c_len = len(c) + 2

            # Find caption lengths

            enc_captions.append(enc_c)
            caplens.append(c_len)
    return enc_captions, caplens


def save_paths(path,img_names):
    """
    quick function to save paths for pegasus
    """
    path = path.split("/")[-1]
    img_names.append(path)

    return img_names

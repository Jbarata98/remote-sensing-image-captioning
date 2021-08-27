from src.captioning_scripts.baseline.base_AttentionModel import Attention
from src.captioning_scripts.pyramid_attention import Channel_Attention,Spatial_Attention
from src.configs.setters.set_initializers import *
import itertools

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PegasusFusionWithAttention(nn.Module):
    """
    Decoder(LSTM) + Pegasus + Soft_Attention
    """

    def __init__(self, aux_lm, aux_dim, attention_dim, embed_dim, decoder_dim, vocab, hashmap, vocab_size, sim_mapping,
                 max_len,
                 attention = ATTENTION,
                 encoder_dim=2048, dropout=0.5):
        """
        :param aux_lm: auxiliary Language Model to fusion with LSTM
        :param aux_dim: auxiliary Language Model dimension (1024)
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab: vocabulary that is being learned
        :param hashmap: conversion hashmap between custom vocab and Transformer's
        :param vocab_size: size of vocabulary
        :param sim_mapping: dictionary with image similarity
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        :param attention : attention type
        """

        super(PegasusFusionWithAttention, self).__init__()

        self.aux_lm = aux_lm

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab = vocab
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.aux_dim = aux_dim
        self.hashmap = hashmap
        self.img_similarity = sim_mapping
        self.max_len = max_len
        self.attention_type = attention
        # if its Soft Attention we're dealing with
        if self.attention_type == ATTENTION_TYPE.soft_attention.value:
            self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network
            self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
            self.sigmoid = nn.Sigmoid()

        # if dealing with pyramid features
        elif self.attention_type == ATTENTION_TYPE.pyramid_attention.value:
            self.channel_attention = Channel_Attention(encoder_dim, decoder_dim, attention_dim)  # attention network
            self.spatial_attention = Spatial_Attention(encoder_dim, decoder_dim, attention_dim)  # attention network

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer

        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell

        self.fc = nn.Linear(decoder_dim + aux_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()

        print("vocab size:", vocab_size)
        print("embed_dim:", embed_dim)
        print("auxLM_dim:", aux_dim)

    def init_weights(self):

        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """

        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):

        """
        Loads embedding layer with pre-trained embeddings.
        :param embeddings: pre-trained embeddings
        """

        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):

        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).
        :param fine_tune: Bool
        """

        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def fine_tune_pegasus(self, fine_tune=False):

        """
        Allow fine-tuning of pegasus
        :param fine_tune: Bool
        """

        for p in self.aux_lm["model"].parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):

        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """

        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def init_pegasus_encoder(self, encoder_input, decoder_input):

        """
        initiates encoder input
        :param encoder_input: the captions from the most similar images
        :param decoder_input: the initializers for decoder input
        :return: the last hidden states for the encoder
        """

        outputs = self.aux_lm["model"](encoder_input.unsqueeze(1), decoder_input_ids=decoder_input, return_dict=True,
                                       output_hidden_states=True)

        encoded_sequence = outputs  # [32,x,1024] 151 if encoder_last_hidden_state

        return encoded_sequence

    def create_pegasus_input(self, pegasus_input, caption_ids):
        """
        Creates the input for pegasus encoder (adds eos token and pads)
        """
        # if dealing with multi inputs on pegasus
        if MULTI_INPUT:
            encoder_input = []
            for pos, img in enumerate(caption_ids):
                encoder_input.append(pegasus_input.get(caption_ids[str(pos + 1)]))

                encoder_input = list(itertools.chain.from_iterable(encoder_input)) + [
                    self.aux_lm["model"].config.eos_token_id]

        else:
            # using only 1 input ( 1 similar image)

            encoder_input = pegasus_input.get(caption_ids) + [self.aux_lm["model"].config.eos_token_id]

        # print("after\n", encoder_input)

        # print("before len encoder input", len(encoder_input))RE

        # print(caption_ids)

        encoder_input = encoder_input + [self.aux_lm["model"].config.pad_token_id] * (self.max_len - len(encoder_input))
        # print("last", encoder_input)
        # print("after len encoder input", len(encoder_input))

        # if len(encoder_input) > 151:
        # print(caption_ids)
        return encoder_input

    def calc_auxLM(self, init_output, decoder_input, bsize_t, t):
        """
        :param init_output: last hidden states from encoder,decoder
        :param decoder_input: input to initialize pegasus decoder
        :param bsize_t: batch_size for that timestep (decreasing lengths)
        :param t: current timestep
        :return: hidden state, cell state
        calculates the hidden state for pegasus
        """

        if t == 0:

            lm_states = init_output.decoder_hidden_states[-1]
            h_prev = lm_states[:, -1:, :].squeeze(1)  # [32,1024]

        else:

            encoder_output = init_output.encoder_last_hidden_state[:bsize_t, :, :]

            outputs_auxLM = self.aux_lm["model"](encoder_outputs=(encoder_output,),
                                                 decoder_input_ids=decoder_input.squeeze(1), return_dict=True,
                                                 output_hidden_states=True)

            auxLM_states = outputs_auxLM.decoder_hidden_states[-1].to(
                device)  # pick the last one, and take only the last hidden state

            h_prev = auxLM_states[:, -1:, :].squeeze(1)  # shape (32,1024)

        return h_prev

    def forward(self, encoder_out, paths, encoded_captions, caption_lengths, pegasus_input):

        """
        Forward propagation.
        :param paths: paths associated with the image
        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size
        # Flatten image if its soft attention, pyramid features already flattened
        if self.attention_type == ATTENTION_TYPE.soft_attention.value:
            encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)

        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_id = caption_lengths.squeeze(1).sort(dim=0, descending=True)

        paths = [paths[sorted_id] for sorted_id in sort_id.tolist()]

        encoder_out = encoder_out[sort_id]
        encoded_captions = encoded_captions[sort_id]

        # Embedding
        # print("encoded_captions shape:", encoded_captions.shape)

        embeddings = self.embedding(encoded_captions).to(device)  # (batch_size, max_caption_length, embed_dim)
        # Initialize LSTM state
        h_lstm, c_lstm = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()
        # initialize the IDs for Pegasus encoder

        # print([self.img_similarity.get(path)['Most similar'] for path in paths])

        encoder_input_ids = torch.LongTensor([self.create_pegasus_input(pegasus_input, self.img_similarity.get(path)[
            'Most similar(s)' if MULTI_INPUT else 'Most similar']) for path in paths]).to(device)

        # initialize tensor for decoder input ids
        aux_lm_ids = torch.LongTensor(
            [[[self.aux_lm["model"].config.decoder_start_token_id]] for _ in range(batch_size)]).to(device)

        # Create tensors to hold word prediction scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        # alphas are for spatial attention only
        if self.attention_type == ATTENTION_TYPE.soft_attention.value:
            alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

        # encoder hidden states for each caption
        pegasus_init_outputs = self.init_pegasus_encoder(encoder_input_ids, aux_lm_ids)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        """----------------------------------------------------------------------------------------------------------"""
        """                                         DECODING PHASE                                                   """
        """----------------------------------------------------------------------------------------------------------"""
        for t in range(max(decode_lengths)):

            # if timestep is not the initial one

            # batch size for that timestep
            batch_size_t = sum([l > t for l in decode_lengths])

            # if its not the first timestep need to concat with previous
            if t > 0:
                # ids_temp = []
                predicted_indexes = torch.LongTensor(next_LM_ids[:batch_size_t]).to(device)
                tokens_tensor = aux_lm_ids[:batch_size_t].to(device)
                LM_cat = torch.cat([tokens_tensor, predicted_indexes], dim=-1)

                aux_lm_ids = LM_cat

                # print("concated")
            # concat with previous ID
            """-------------------------------- ATTENTION COMPUTATION----------------------------------------------"""
            if self.attention_type == ATTENTION_TYPE.soft_attention.value:
                # if soft attention calculate sigmoid gate with soft attention and store alphas to calculate the loss
                attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                    h_lstm[:batch_size_t])

                gate = self.sigmoid(self.f_beta(h_lstm[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
                attention_weighted_encoding = gate * attention_weighted_encoding

            # if its pyramid attention need to calculate channel and spatial attention
            if self.attention_type == ATTENTION_TYPE.pyramid_attention.value:
                v_s, _ = self.spatial_attention(encoder_out[:batch_size_t],
                                                                    h_lstm[:batch_size_t])
                v_c= self.channel_attention(encoder_out[:batch_size_t],
                                                                h_lstm[:batch_size_t])

                # sum both attentions
                attention_weighted_encoding = v_s + v_c

            """------------------------------------------------------------------------------------------------------"""
            # LSTM
            # print("attention weighted")
            """------------------------------------- DECODE STEP-----------------------------------------------------"""
            # calculate hidden state for Pegasus
            h_auxLM = self.calc_auxLM(pegasus_init_outputs, aux_lm_ids, batch_size_t, t)
            # print("hidden_state_calculated")
            h_lstm, c_lstm = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h_lstm[:batch_size_t], c_lstm[:batch_size_t]))  # (batch_size_t, decoder_dim)

            """------------------------------------------------------------------------------------------------------"""
            # print("decode step")

            """----------------------------------------- FUSION -----------------------------------------------------"""
            # simple fusion
            h_fusion = torch.cat([h_lstm, h_auxLM], axis=-1)
            # print('fusion success')

            """------------------------------------------------------------------------------------------------------"""

            """-------------------------------------------PREDICTION-------------------------------------------------"""
            # calculate predictions
            preds = self.fc(self.dropout(h_fusion))  # (batch_size_t, vocab_size)
            # print("got_preds")
            predictions[:batch_size_t, t, :] = preds
            if self.attention_type == ATTENTION_TYPE.soft_attention.value:
                alphas[:batch_size_t, t, :] = alpha

            # next IDs for the pegasus
            next_LM_ids = torch.argmax(preds, dim=-1).to(device)  #
            # print("max_ids")
            # # if using custom vocabulary need to convert before passing it on to pegasus
            if CUSTOM_VOCAB:
                next_LM_ids = [[[self.hashmap.get(str(x.item()))]] for x in next_LM_ids]

            # no need for conversion if using the same vocab
            else:
                next_LM_ids = [[[x]] for x in next_LM_ids]

            # concat the ids(previous word with current word)
            # print("got new ids")
            """------------------------------------------------------------------------------------------------------"""
        if self.attention_type == ATTENTION_TYPE.soft_attention.value:
            return predictions, encoded_captions, decode_lengths, alphas, sort_id
        if self.attention_type == ATTENTION_TYPE.pyramid_attention.value:
            return predictions, encoded_captions, decode_lengths, sort_id

#

from src.configs.setters.set_initializers import *
from src.captioning_scripts.baseline.base_AttentionModel import Attention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




class GPT2FusionWithAttention(nn.Module):
    """
    Decoder + GPT2 + Attention
    """

    def __init__(self, aux_lm, aux_dim, attention_dim, embed_dim, decoder_dim, vocab, hashmap, vocab_size,
                 encoder_dim=2048,
                 dropout=0.5):

        """
        :param auxLM: auxiliary Language Model to fusion with LSTM
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """

        super(GPT2FusionWithAttention, self).__init__()

        self.aux_LM = aux_lm["model"]

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab = vocab
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.aux_dim = aux_dim
        self.hashmap = hashmap
        self.tokenizer = aux_lm["tokenizer"]

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network
        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer

        print("vocab size:", vocab_size)
        print("embed_dim:", embed_dim)
        print("auxLM_dim:", aux_dim)

        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim + aux_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

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
        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def fine_tune_gpt2(self, fine_tune=False):
        """
        Allow fine-tuning of gpt2?
        :param fine_tune: Allow?
        """
        for p in self.aux_LM.parameters():
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

    # calculate  hidden states for auxLM
    def calc_auxLM(self, ids, bsize_t, t, eval=False):

        h_prev = torch.zeros(bsize_t, self.aux_dim).to(
            device)  # initialize a list to save the hidden states with batch-size for that timestep
        subbatch_size = 4
        # divide ids into sublist of ids for faster processing ( batch/2)
        new_ids = [ids[i:i + subbatch_size] for i in range(0, len(ids), subbatch_size)]

        if t == 0:
            aux_counter = 0
            for i, id in enumerate(new_ids):

                # first word

                outputs_auxLM = self.aux_LM(id.squeeze(1), return_dict=True, output_hidden_states=True)
                auxLM_states = outputs_auxLM.hidden_states[-1].to(device)

                for index, h_state in enumerate(auxLM_states):
                    h_prev[index + aux_counter] = h_state  # (1,1,768)
                aux_counter += subbatch_size

            # stack works because in t=0 they are all same size(batch_size)

            return h_prev

        else:
            aux_counter = 0
            for i, id in enumerate(new_ids):
                # remaining timesteps

                # input = torch.LongTensor([id.item()])
                input = id

                outputs_auxLM = self.aux_LM(input, return_dict=True, output_hidden_states=True)
                auxLM_states = outputs_auxLM.hidden_states[-1].to(
                    device)  # pick the last one, and take only the last hidden state


                # each value in the batch
                for i, h_state in enumerate(auxLM_states):
                    h_prev[i + aux_counter] = h_state[:, -1:, :]  # (1,1,768)
                aux_counter += subbatch_size

        return h_prev

    def forward(self, encoder_out, encoded_captions, caption_lengths):

        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        # print("encoded_captions shape:", encoded_captions.shape)
        embeddings = self.embedding(encoded_captions).to(device)  # (batch_size, max_caption_length, embed_dim)
        # Initialize LSTM state
        h_lstm, c_lstm = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)
        # init auxLM

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # initialize the IDs for language model ( bos token) * batch_size
        LM_ids = torch.LongTensor([[[self.tokenizer.bos_token_id]] for _ in range(batch_size)]).to(device)

        # Create tensors to hold word prediction scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding

        for t in range(max(decode_lengths)):

            # if timestep is not the initial one

            # batch size for that timestep
            batch_size_t = sum([l > t for l in decode_lengths])

            # if its not the first timestep need to concat with previous
            if t > 0:
                # ids_temp = []
                predicted_indexes = torch.LongTensor(next_LM_ids[:batch_size_t]).to(device)
                tokens_tensor = LM_ids[:batch_size_t].to(device)
                LM_cat = torch.cat([tokens_tensor, predicted_indexes], dim=-1)

                LM_ids = LM_cat
                # print("concated")
            # concat with previous ID

            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h_lstm[:batch_size_t])

            gate = self.sigmoid(self.f_beta(h_lstm[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            # LSTM

            h_auxLM = self.calc_auxLM(LM_ids, batch_size_t, t)
            # print(" calculated auxLM")

            h_lstm, c_lstm = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h_lstm[:batch_size_t], c_lstm[:batch_size_t]))  # (batch_size_t, decoder_dim)

            # print("decode step")

            # simple fusion
            h_fusion = torch.cat([h_lstm, h_auxLM], axis=-1)
            # print('fused')
            # calculte predictions
            preds = self.fc(self.dropout(h_fusion))  # (batch_size_t, vocab_size)
            # print("got_preds")
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

            # next IDs for the gpt2
            next_LM_ids = torch.argmax(preds, dim=-1).to(device)  #
            # print("max_ids")
            # # if using custom vocabulary need to convert before passing it on to gpt2
            if CUSTOM_VOCAB:

                next_LM_ids = [[[self.hashmap.get(str(x.item()))]] for x in next_LM_ids]


            # no need for conversion if using the same vocab
            else:
                next_LM_ids = [[[x]] for x in next_LM_ids]
            # concat the ids(previous word with current word)
            # print("got new ids")
        # print("decoded")
        return predictions, encoded_captions, decode_lengths, alphas, sort_ind

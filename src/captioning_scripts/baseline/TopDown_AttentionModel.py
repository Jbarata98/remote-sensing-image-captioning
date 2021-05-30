from src.configs.setters.set_initializers import *
from src.captioning_scripts.baseline.base_AttentionModel import Attention, LSTMWithAttention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMWithTopDownAttention(LSTMWithAttention):
    """
    Decoder with TopDown Attention (Double LSTM).
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout = 0.5):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """

        super(LSTMWithTopDownAttention, self).__init__(attention_dim,embed_dim,decoder_dim,vocab_size,encoder_dim,dropout)

        self.LanguageLSTM = nn.LSTMCell(decoder_dim + encoder_dim, decoder_dim, bias=True)  # decoding Language LSTM
        self.TopDownLSTM = nn.LSTMCell(embed_dim + encoder_dim + decoder_dim, decoder_dim, bias=True)  # TopDown_Attention

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
        # print("encoded_captions length:", encoded_captions.shape)
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        h_topdown, c_topdown = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)
        h_language, c_language = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

        # mean-pooled image feature for input to the topdown lstm
        encoder_out_mean_pool = encoder_out.mean(dim=1)


        # At each time-step, decode by

        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])

            # The input vector to the attention LSTM at each time step consists of the previous output of the language LSTM, concatenated with the mean-pooled image feature
            #  an encoding of the previously generated word ( in this case of teacher forcing its the real word)

            h_topdown, c_topdown = self.TopDownLSTM(
                torch.cat([embeddings[:batch_size_t, t, :],encoder_out_mean_pool[:batch_size_t],h_language[:batch_size_t]], dim=1),
                (h_topdown[:batch_size_t], c_topdown[:batch_size_t]))  # (batch_size_t, decoder_dim)

            # attention-weighing the encoder's output based on the decoder's previous hidden state output
            # then generate a new word in the decoder with the previous word and the attention weighted encoding

            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h_topdown[:batch_size_t])

            gate = self.sigmoid(self.f_beta(h_topdown[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
        #
            h_language,c_language = self.LanguageLSTM(torch.cat([h_topdown[:batch_size_t], attention_weighted_encoding], dim = 1),
                                    (h_language[:batch_size_t], c_language[:batch_size_t]))

            preds = self.fc(self.dropout(h_language))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha
        #
        return predictions, encoded_captions, decode_lengths, alphas, sort_ind

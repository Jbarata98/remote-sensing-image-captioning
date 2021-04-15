import torch
from torch import nn
import torchvision
from configs.initializers import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """

        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """

        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)

        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)
        return attention_weighted_encoding, alpha



class FusionWithAttention(nn.Module):
    """
    Decoder.
    """

    def __init__(self, auxLM, aux_dim, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048,
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

        super(FusionWithAttention, self).__init__()

        self.aux_LM = auxLM
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.aux_dim = aux_dim

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network
        print("vocab size:", vocab_size)
        print("embed_dim:", embed_dim)
        print("auxLM_dim:", aux_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer

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
    def calc_auxLM(self, ids,decode_lengths, bsize_t, t):

        h_prev = torch.zeros(bsize_t, self.aux_dim).to(device) # initialize a list to save the hidden states with batch-size for that timestep

        if t == 0:
            for i,id in enumerate(ids):

                # first word
                outputs_auxLM = self.aux_LM(id, return_dict=True, output_hidden_states=True)
                auxLM_states = outputs_auxLM.hidden_states[-1]

                h_prev[i] = auxLM_states #(1,1,768)

            #stack works because in t=0 they are all same size(batch_size)

            return h_prev

        else:

            for i,id in enumerate(ids):
                # remaining timesteps

                # input = torch.LongTensor([id.item()])
                input = id

                outputs_auxLM = self.aux_LM(input, return_dict=True, output_hidden_states=True)
                auxLM_states = outputs_auxLM.hidden_states[-1] #pick the last one, and take only the last hidden state


                h_prev[i] = auxLM_states[:,-1:,:] #(1,1,768)


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
        LM_ids = torch.LongTensor([[[AuxLM_tokenizer.bos_token_id]] for _ in range(batch_size)])

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

            #if its not the first timestep need to concat with previous
            if t >0:
                # ids_temp = []

                # for it in range(batch_size_t):
                #     #concat current with previous
                LM_cat = torch.cat([torch.LongTensor(LM_ids[:batch_size_t]), torch.LongTensor(next_LM_ids[:batch_size_t])],dim=-1)
                #                       dim=-1)  # next_LM_ids#]
                #     ids_temp.append(LM_id)
                LM_ids = LM_cat
            # concat with previous ID

            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h_lstm[:batch_size_t])

            gate = self.sigmoid(self.f_beta(h_lstm[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            # LSTM

            h_auxLM = self.calc_auxLM(LM_ids,decode_lengths, batch_size_t, t)


            h_lstm, c_lstm = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h_lstm[:batch_size_t], c_lstm[:batch_size_t]))  # (batch_size_t, decoder_dim)


            # print("h_auxlms:",h_auxLM.shape)
            # print("h_lstm:",h_lstm.shape)


            # simple fusion
            h_fusion = torch.cat([h_lstm, h_auxLM], axis=-1)

            # calculte predictions
            preds = self.fc(self.dropout(h_fusion))  # (batch_size_t, vocab_size)

            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

            # next IDs for the gpt2
            next_LM_ids = torch.argmax(preds, dim=-1)  # (batch_size)
            next_LM_ids = [[[x]] for x in next_LM_ids]
            #concat the ids(previous word with current word)



        # print("decoded")
        return predictions, encoded_captions, decode_lengths, alphas, sort_ind
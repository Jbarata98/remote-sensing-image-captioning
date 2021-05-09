from src.captioning_scripts.fusion.gpt2.decoder_gpt2_simple import Attention
from src.configs.setters.set_initializers import *

class PegasusFusionWithAttention(nn.Module):
    """
    Decoder + Pegasus + Soft_Attention
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

        super(PegasusFusionWithAttention, self).__init__()

        self.aux_LM = aux_lm

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab = vocab
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.aux_dim = aux_dim
        self.hashmap = hashmap

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
        self.init_weights()

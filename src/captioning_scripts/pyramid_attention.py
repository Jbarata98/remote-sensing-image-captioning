from src.configs.setters.set_initializers import *
from src.captioning_scripts.abstract_encoder import Encoder


class Spatial_Attention(nn.Module):
    """
    Spatial Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """

        super(Spatial_Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """

        # print("encoder_out", encoder_out.shape, "decoder_hidden", decoder_hidden.shape)
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)

        att = self.full_att(self.tanh(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)

        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)
        return attention_weighted_encoding, alpha

class Channel_Attention(nn.Module):
    """
    Channel Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """

        super(Channel_Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim,encoder_dim)  # linear layer to calculate values to be sigmoid
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """

        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)

        att = self.full_att(self.tanh(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        beta = self.sigmoid(att)  # (batch_size, num_pixels)

        attention_weighted_encoding = beta.mean(dim=1).squeeze(1) * encoder_out.sum(dim=1) # (batch_size, encoder_dim)
        if VISUALIZATION:
            return attention_weighted_encoding, beta
        else:
            return attention_weighted_encoding


#testing code
# encoder = Encoder(model_type=ENCODER_MODEL, pyramid_kernels=[(1,1),(2,2),(4,4)] ,fine_tune=FINE_TUNED_PATH)
#
# img = torch.randn(1,3,224,224)
#
# decoder_hidden = torch.randn(1,512)
#
# att = Spatial_Attention(encoder_dim = 2048, decoder_dim=512, attention_dim=512)
# dual_att = Channel_Attention(encoder_dim = 2048, decoder_dim=512, attention_dim=512)
#
# v_s = att(encoder(img),decoder_hidden)[0]
# v_c = dual_att(encoder(img),decoder_hidden)[0]
#
# v_dual = v_s + v_c
# print(v_dual.shape)



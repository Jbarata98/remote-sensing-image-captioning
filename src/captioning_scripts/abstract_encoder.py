import logging
from torch import nn

from src.configs.globals import ENCODERS
from src.configs.setters.set_initializers import *


ENCODER = Setters()._set_encoder()

class Encoder(nn.Module):
    """
    Encoder.
    """
    def __init__(self, model_type, encoded_image_size=14, fine_tune = False):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size
        self.encoder_model = model_type  # pretrained ImageNet model
        self.model, self.encoder_dim = ENCODER._get_encoder_model()
        print("dimension of encoder:", self.encoder_dim)

        # # Remove linear and pool layers (since we're not doing classification)
        # modules = list(encoder_model.children())[:-2]
        # self.encoder_model = nn.Sequential(*modules)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        for p in self.model.parameters():
            p.requires_grad = False

        self.fine_tune(fine_tune)

    def forward(self, images):
        """
        Forward propagation.
        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        # out = self.encoder_model(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.model.extract_features(images)

        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)

        return out

    def fine_tune(self, fine_tune=True):

        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.
        :param fine_tune: Allow?
        """

        print("Fine-tune encoder:", fine_tune)

        # If fine-tuning
        if self.encoder_model == ENCODERS.EFFICIENT_NET_IMAGENET.value: #base model to fine tune #do it in the end for last test
            logging.info("Fine tuning base model...")
            for c in list(self.model.children()): #all layers
                for p in c.parameters():
                    p.requires_grad = fine_tune

        elif self.encoder_model == ENCODERS.EFFICIENT_NET_IMAGENET_FINETUNED.value: #already finetuned
            logging.info("Loading already fine-tuned...")
            for c in list(self.model.children()): #all layers
                for p in c.parameters():
                    p.requires_grad = fine_tune


        #todo rest of captioning_scripts
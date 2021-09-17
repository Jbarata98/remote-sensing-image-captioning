from src.configs.setters.set_initializers import *

# ENCODER = Setters('../configs/setters/training_details.txt')._set_encoder()


ENCODER = Setters()._set_encoder()


class Encoder(nn.Module):
    """
    Encoder.
    """

    def __init__(self, model_type, model_version = 'v1', pyramid_kernels=[], encoded_image_size=14, fine_tune=False):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size
        self.eff_net_version = model_version
        self.pyramid_kernels = pyramid_kernels
        self.encoder_model = model_type  # pretrained ImageNet model
        self.model, self.encoder_dim = ENCODER._get_encoder_model(eff_net_version=model_version)
        logging.info("dimension of encoder: {}".format(self.encoder_dim))

        # # Remove linear and pool layers (since we're not doing classification)
        # modules = list(encoder_model.children())[:-2]
        # self.encoder_model = nn.Sequential(*modules)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        self.avg_pool = nn.AvgPool2d

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

        if self.eff_net_version == 'v1':
            out = self.model.extract_features(images)

        if self.eff_net_version == 'v2':
            # eff net v2 has a different method to extract featuresbh
            out = self.model.forward_features(images)

        # if using soft attention, only need one final pooling over the results
        if ATTENTION == ATTENTION_TYPE.soft_attention.value:
            out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size) #1280 if V2
            out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048) #1280 if V2

        # pyramid attention, performs pyramid feature maps with diff pooling
        elif ATTENTION == ATTENTION_TYPE.pyramid_attention.value:
            pyramid_feature_maps = []
            for kernel in self.pyramid_kernels:
                # print(out.shape)
                pyramid_feature_maps.append(
                    self.avg_pool(kernel_size=kernel, stride=1)(out).permute(0, 2, 3, 1).flatten(start_dim=1,end_dim=2))
            # reshape and concat first 3  (batch_size,bins_1+bins_2+bins_3,2048)
            # print(len(pyramid_feature_maps))

        return out

    def fine_tune(self, fine_tune=True):

        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.
        :param fine_tune: Allow?
        """

        logging.info("Fine-tune encoder: {}".format(fine_tune))

        # If fine-tuning
        if self.encoder_model == ENCODERS.EFFICIENT_NET_IMAGENET.value or self.encoder_model == ENCODERS.EFFICIENT_NET_V2_IMAGENET.value:  # base model to fine tune #do it in the end for last test
            logging.info("Fine tuning base model...")
            for c in list(self.model.children()):  # all layers
                for p in c.parameters():
                    p.requires_grad = fine_tune

        elif self.encoder_model == ENCODERS.EFFICIENT_NET_IMAGENET_FINETUNED.value:  # already finetuned
            logging.info("Loading already fine-tuned...")
            for c in list(self.model.children()):  # all layers
                for p in c.parameters():
                    p.requires_grad = fine_tune


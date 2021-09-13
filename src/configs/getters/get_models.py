import json
import logging
import os
import torch
import timm
from src.configs.setters.set_enums import ENCODERS, AUX_LMs
from src.configs.globals import LOSS, LOSSES, TASK
from src.classification_scripts.SupConLoss.SupConModel import SupConEffNet

from torchvision import models
from efficientnet_pytorch import EfficientNet
from transformers import (PegasusForConditionalGeneration,
                          PegasusTokenizer,
                          PegasusConfig,
                          GPT2Tokenizer,
                          GPT2Config,
                          GPT2LMHeadModel)

log = logging.getLogger()  # 'root' Logger

console = logging.StreamHandler()

format_str = '%(asctime)s\t%(levelname)s -- %(processName)s %(filename)s:%(lineno)s -- %(message)s'
console.setFormatter(logging.Formatter(format_str))

log.addHandler(console)  # prints to console.

logging.root.setLevel(logging.INFO)


# -----------------------------------MODELS-----------------------------------------------
class GetEncoders:
    """
    Class to get the model encoder
    """

    def __init__(self, model, checkpoint_path=None, device=None, nr_classes=31):
        self.model = model
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.nr_classes = nr_classes

    # only using resnet and eff_net for tests
    def _get_encoder_model(self, eff_net_version = 'v1'):
        self.eff_net_version = eff_net_version

        logging.info('EFFICIENT_NET VERSION: {}'.format(self.eff_net_version))

        if self.model == ENCODERS.RESNET.value:
            logging.info("image model with resnet model")

            image_model = models.resnet101(pretrained=True)
            # modules = list(image_model.children())[:-2]
            encoder_dim = 2048

            return image_model, encoder_dim


        # load from checkpoint the encoder #eff_net
        else:
            if self.model == ENCODERS.EFFICIENT_NET_IMAGENET_FINETUNED.value:
                logging.info("using image model with efficientnet-b5 model pre-trained on RSICD")
                image_model = EfficientNet.from_pretrained('efficientnet-b5', num_classes=self.nr_classes)
                encoder_dim = image_model._fc.in_features

            elif self.model == ENCODERS.EFFICIENT_NET_IMAGENET_FINETUNED_AUGMENTED.value:
                logging.info("using image model with efficientnet-b5 model pre-trained and transformations on RSICD")
                image_model = EfficientNet.from_pretrained('efficientnet-b5', num_classes=self.nr_classes)
                encoder_dim = image_model._fc.in_features

            elif self.model == ENCODERS.EFFICIENT_NET_IMAGENET_FINETUNED_AUGMENTED_CONTRASTIVE.value:
                # contrastive doesnt use last layer with diff nr of classes

                logging.info(
                    "using image model with efficientnet-b5 model pre-trained and transformations and Supervised Contrastive Loss on RSICD")
                image_model = SupConEffNet(eff_net_version='v1')
                encoder_dim = image_model.encoder_dim


            elif self.model == ENCODERS.EFFICIENT_NET_IMAGENET.value:
                # https://github.com/lukemelas/EfficientNet-PyTorch/pull/194
                image_model = EfficientNet.from_pretrained('efficientnet-b5', num_classes=self.nr_classes)
                encoder_dim = image_model._fc.in_features

            elif self.model == ENCODERS.EFFICIENT_NET_V2_IMAGENET.value:
                logging.info("image model with efficientnet_v2_medium model pre-trained on imagenet")
                image_model = timm.create_model('tf_efficientnetv2_m_in21k',pretrained=True)
                encoder_dim = image_model.forward_features(torch.randn(1,3,224,224)).shape[1] #1280

            elif self.model == ENCODERS.EFFICIENT_NET_V2_IMAGENET_FINETUNED_AUGMENTED_CONTRASTIVE.value:
                logging.info("image model with efficientnet_v2_medium model pre-trained on imagenet with augmentations and SupConLoss")
                image_model = SupConEffNet(eff_net_version=self.eff_net_version)
                # image_model = supconeffv2.model
                encoder_dim = image_model.encoder_dim

            elif self.model == ENCODERS.EFFICIENT_NET_V2_IMAGENET_FINETUNED_AUGMENTED_CONTRASTIVE_CE.value:
                logging.info("image model with efficientnet_v2_medium model pre-trained on imagenet with augmentations, SupConLoss & extra epochs w/ CE")
                image_model = SupConEffNet(eff_net_version=self.eff_net_version)
                encoder_dim = image_model.encoder_dim

            elif self.model == ENCODERS.EFFICIENT_NET_V2_IMAGENET_FINETUNED_AUGMENTED_CONTRASTIVE_ALS.value:
                logging.info("image model with efficientnet_v2_medium model pre-trained on imagenet with augmentations, SupConLoss & extra epochs w/ ALS")
                image_model = SupConEffNet(eff_net_version=self.eff_net_version)
                encoder_dim = image_model.encoder_dim

            else:
                logging.info("unsupported model, quitting...")
                exit()

            # load the checkpoint
            # import sys
            # print(sys.path)
            # print(self.checkpoint_path)
            if os.path.exists(self.checkpoint_path):
                logging.info("loading pretrained encoder in {}...".format(self.checkpoint_path))
                if torch.cuda.is_available():
                    logging.info("Device: {}".format(self.device))
                    checkpoint = torch.load(self.checkpoint_path)

                else:
                    logging.info("Device: {}".format(self.device))
                    checkpoint = torch.load(self.checkpoint_path, map_location=torch.device('cpu'))

                # nr of classes for RSICD
                # image_model._fc = nn.Linear(encoder_dim, output_layer_size)

                if TASK == 'Captioning' or TASK == 'Retrieval':
                    # returns the weights already
                    image_model.load_state_dict(checkpoint['model'])
                    if self.model == ENCODERS.EFFICIENT_NET_IMAGENET_FINETUNED_AUGMENTED_CONTRASTIVE.value:
                        image_model = image_model.model
                    elif self.model == ENCODERS.EFFICIENT_NET_V2_IMAGENET_FINETUNED_AUGMENTED_CONTRASTIVE.value:
                        image_model = image_model.model
                    elif self.model == ENCODERS.EFFICIENT_NET_V2_IMAGENET_FINETUNED_AUGMENTED_CONTRASTIVE_CE.value:
                        image_model = image_model.model

                    return image_model, encoder_dim

                elif TASK =='Classification':
                    # returns the full model
                    return image_model, encoder_dim

            # pretrained encoder checkpoint doesn't exist - for baseline/classification pretraining
            else:
                logging.info("pretrained encoder path does not exist, continuing...")

                if LOSS == LOSSES.Cross_Entropy.value:
                    logging.info("setting up model for cross_entropy...")

                    if self.eff_net_version == 'v1':
                        encoder_dim = image_model._fc.in_features
                    elif self.eff_net_version =='v2':
                        encoder_dim = image_model.model.forward_features(torch.randn(1, 3, 224, 224)).shape[1]  # 1280

                    return image_model, encoder_dim
                elif LOSS == LOSSES.SupConLoss.value:
                    # print(self.nr_classes)
                    # alter the nr of classes for transfer learning
                    image_model = SupConEffNet(eff_net_version=self.eff_net_version)

                    return image_model, encoder_dim


class GetAuxLM:
    """
    Class to get the model AuxLM(s)
    """

    def __init__(self, model, checkpoint_path=None, device=None):

        self.model = model
        self.checkpoint_path = checkpoint_path
        self.device = device

    def _get_decoder_model(self, special_tokens=None, pretrained=False):

        if self.model == AUX_LMs.GPT2.value:

            logging.info("loading GPT2 model...")

            model_name = 'gpt2'  # use small
            tokenizer = GPT2Tokenizer.from_pretrained(model_name)

            if special_tokens:

                logging.info("Adding special tokens to tokenizer...")
                tokenizer.add_special_tokens(special_tokens)

                logging.info("Adding special tokens to GPT2...")
                config = GPT2Config.from_pretrained(model_name,
                                                    bos_token_id=tokenizer.bos_token_id,
                                                    eos_token_id=tokenizer.eos_token_id,
                                                    pad_token_id=tokenizer.pad_token_id,
                                                    unk_token=tokenizer.unk_token_id,
                                                    output_hidden_states=True)

            else:
                config = GPT2Config.from_pretrained(model_name,
                                                    pad_token_id=tokenizer.eos_token_id,
                                                    output_hidden_states=True)

            model = GPT2LMHeadModel.from_pretrained(model_name, config=config).to(self.device)

            if special_tokens:  # Special tokens added, model needs to be resized accordingly
                logging.info("resized accordingly...")
                model.resize_token_embeddings(len(tokenizer))

            print("size of gpt2 vocab:", len(tokenizer))
            # print(tokenizer.all_special_tokens)
            # print(tokenizer.unk_token)
            # print("id:", tokenizer.unk_token_id)

            return tokenizer, model

        elif self.model == AUX_LMs.PEGASUS.value:

            if pretrained:

                logging.info("loading PRETRAINED Pegasus model...")

                # change root path depending on where is the model in your local environment
                ## LOCAL
                # model_name = '/home/starksultana/Documentos/MEIC/5o_ano/Tese/code/remote-sensing-image-captioning/experiments/fusion/fine_tuned/checkpoints/pegasus/checkpoint_pretrain_pegasus/model_xsum_similar_captions_pretrain'
                ## REMOTE
                model_name = '/home/guests/jmb/experiments/fusion/fine_tuned/checkpoints/pegasus/checkpoint_pretrain_pegasus/model_xsum_similar_captions_pretrain/'

                tokenizer = PegasusTokenizer.from_pretrained(model_name)

                if special_tokens:
                    logging.info("Adding special tokens to Pegasus...")
                    tokenizer.add_special_tokens(special_tokens)
                    config = PegasusConfig.from_pretrained(model_name,
                                                           bos_token_id=tokenizer.bos_token_id,
                                                           eos_token_id=tokenizer.eos_token_id,
                                                           pad_token_id=tokenizer.pad_token_id,
                                                           unk_token=tokenizer.unk_token_id,
                                                           output_hidden_states=True)

                else:
                    config = PegasusConfig.from_pretrained(model_name,
                                                           output_hidden_states=True)

                model = PegasusForConditionalGeneration.from_pretrained(model_name, config=config).to(self.device)
                logging.info("loaded PRETRAINED Pegasus model...")

                # to use auxLM pretrained on local data
            else:
                logging.info("loading Pegasus model...")

                model_name = 'google/pegasus-large'  # fixed for extractive summary only

                tokenizer = PegasusTokenizer.from_pretrained(model_name)

                if special_tokens:
                    logging.info("Adding special tokens to Pegasus...")
                    tokenizer.add_special_tokens(special_tokens)
                    config = PegasusConfig.from_pretrained(model_name,
                                                           bos_token_id=tokenizer.bos_token_id,
                                                           eos_token_id=tokenizer.eos_token_id,
                                                           pad_token_id=tokenizer.pad_token_id,
                                                           unk_token=tokenizer.unk_token_id,
                                                           output_hidden_states=True)
                else:
                    config = PegasusConfig.from_pretrained(model_name,
                                                           output_hidden_states=True)

                model = PegasusForConditionalGeneration.from_pretrained(model_name, config=config).to(self.device)

                logging.info("loaded Pegasus model...")

            return tokenizer, model


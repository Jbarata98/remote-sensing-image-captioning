import json
import logging
import os

import torch
from src.configs.setters.set_enums import ENCODERS, AUX_LMs
from src.configs.globals import LOSS,LOSSES
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
    def _get_encoder_model(self):

        if self.model == ENCODERS.RESNET.value:
            logging.info("image model with resnet model")

            image_model = models.resnet101(pretrained=True)
            # modules = list(image_model.children())[:-2]
            encoder_dim = 2048

            return image_model, encoder_dim

        # load from checkpoint the encoder
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
                image_model = SupConEffNet()
                encoder_dim = image_model.encoder_dim

            elif self.model == ENCODERS.EFFICIENT_NET_IMAGENET.value:
                # https://github.com/lukemelas/EfficientNet-PyTorch/pull/194
                logging.info("image model with efficientnet-b5 model pre-trained on imagenet")
                image_model = EfficientNet.from_pretrained('efficientnet-b5', num_classes = self.nr_classes)
                encoder_dim = image_model._fc.in_features

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
                    checkpoint = torch.load( self.checkpoint_path)

                else:
                    logging.info("Device: {}".format(self.device))
                    checkpoint = torch.load(self.checkpoint_path, map_location=torch.device('cpu'))



                # nr of classes for RSICD
                # image_model._fc = nn.Linear(encoder_dim, output_layer_size)
                image_model.load_state_dict(checkpoint['model'])
                if self.model == ENCODERS.EFFICIENT_NET_IMAGENET_FINETUNED_AUGMENTED_CONTRASTIVE.value:
                    image_model = image_model.model

                return image_model, encoder_dim

            # pretrained encoder checkpoint doesn't exist - for baseline/classification pretraining
            else:
                logging.info("pretrained encoder path does not exist, continuing...")

                if LOSS == LOSSES.Cross_Entropy.value:
                    logging.info("setting up pretrained model for cross_entropy...")

                    encoder_dim = image_model._fc.in_features

                    return image_model, encoder_dim
                elif LOSS == LOSSES.SupConLoss.value:
                    # print(self.nr_classes)
                    # alter the nr of classes for transfer learning
                    image_model = SupConEffNet()

                    return image_model, image_model.encoder_dim


class GetAuxLM:
    """
    Class to get the model AuxLM(s)
    """

    def __init__(self, model, checkpoint_path=None, device=None):

        self.model = model
        self.checkpoint_path = checkpoint_path
        self.device = device
    def _get_decoder_model(self, special_tokens=None, pretrained = False):

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
            logging.info("loading Pegasus model...")

            if pretrained:


                model_name = 'google/pegasus-xsum'  # fixed for extractive summary only

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

                model = PegasusForConditionalGeneration.from_pretrained(model_name, config = config).to(self.device)

                logging.info("loading PRETRAINED Pegasus model...")

                tokenizer.save_pretrained('../../experiments/fusion/fine_tuned/checkpoints/pegasus/checkpoint_pretrain_pegasus')
                model.save_pretrained('.../../experiments/fusion/fine_tuned/checkpoints/pegasus/checkpoint_pretrain_pegasus')


                tokenizer = PegasusTokenizer.from_pretrained('../../experiments/fusion/fine_tuned/checkpoints/pegasus/checkpoint_pretrain_pegasus')

                from os import listdir
                from os.path import isfile, join
                onlyfiles = [f for f in listdir("../../experiments/fusion/fine_tuned/checkpoints/pegasus/checkpoint_pretrain_pegasus") if isfile(join("../../experiments/fusion/fine_tuned/checkpoints/pegasus/checkpoint_pretrain_pegasus", f))]
                print(onlyfiles)
                # config = PegasusConfig.from_pretrained("../../experiments/fusion/fine_tuned/checkpoints/pegasus/checkpoint_pretrain_pegasus/")
                model = PegasusForConditionalGeneration.from_pretrained('../../experiments/fusion/fine_tuned/checkpoints/pegasus/checkpoint_pretrain_pegasus').to(self.device)

                # to use auxLM pretrained on local data

            else:

                model_name = 'google/pegasus-xsum'  # fixed for extractive summary only

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





            return tokenizer, model

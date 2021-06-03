import logging
import os

import torch
from torch import nn
from src.configs.setters.set_enums import ENCODERS, AUX_LMs

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

class Encoders:
    """
    Class to get the model encoder
    """

    def __init__(self, model, checkpoint_path=None, device=None, nr_classes=31):
        self.model = model
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.nr_classes = nr_classes

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
            elif self.model == ENCODERS.EFFICIENT_NET_IMAGENET_FINETUNED_AUGMENTED.value:
                logging.info("using image model with efficientnet-b5 model pre-trained and transformations on RSICD")
            elif self.model == ENCODERS.EFFICIENT_NET_IMAGENET_FINETUNED_AUGMENTED_CONTRASTIVE.value:
                logging.info(
                    "using image model with efficientnet-b5 model pre-trained and transformations and Supervised Contrastive Loss on RSICD")
            elif self.model == ENCODERS.EFFICIENT_NET_IMAGENET.value:
                # https://github.com/lukemelas/EfficientNet-PyTorch/pull/194
                logging.info("image model with efficientnet-b5 model pre-trained on imagenet")
            else:
                logging.info("unsupported model, quitting...")
                exit()

            # load the checkpoint
            if os.path.exists(self.checkpoint_path):
                logging.info("loading pretrained encoder in {}...".format(self.checkpoint_path))
                if torch.cuda.is_available():
                    logging.info("Device: {}".format(self.device))
                    checkpoint = torch.load( self.checkpoint_path)

                else:
                    logging.info("Device: {}".format(self.device))
                    checkpoint = torch.load(self.checkpoint_path, map_location=torch.device('cpu'))

                image_model = EfficientNet.from_pretrained('efficientnet-b5', num_classes=self.nr_classes)
                encoder_dim = image_model._fc.in_features

                # nr of classes for RSICD
                # image_model._fc = nn.Linear(encoder_dim, output_layer_size)
                image_model.load_state_dict(checkpoint['model'])
                return image_model, encoder_dim

            else:
                logging.info("pretrained encoder path does not exist, continuing...")
                # print(self.nr_classes)

                image_model = EfficientNet.from_pretrained('efficientnet-b5')
                encoder_dim = image_model._fc.in_features

                return image_model, encoder_dim


class AuxLM:
    """
    Class to get the model AuxLM(s)
    """

    def __init__(self, model, checkpoint_path=None, device=None):

        self.model = model
        self.checkpoint_path = checkpoint_path
        self.device = device

    def _get_decoder_model(self, special_tokens=None):

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

        if self.model == AUX_LMs.PEGASUS.value:
            logging.info("loading Pegasus model...")

            model_name = 'google/pegasus-xsum'  # fixed for extractive summary only

            tokenizer = PegasusTokenizer.from_pretrained(model_name)

            # logging.info("Adding special tokens to tokenizer...")

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

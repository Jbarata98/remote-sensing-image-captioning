import torch.optim
import torch.utils.data

from src.configs.setters.set_initializers import *

cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead


class AbstractEvaluator:

    def __init__(self, encoder, decoder, device, checkpoint, b_size=3):
        # Parameters
        # sets device for model and PyTorch tensors

        self.device = device
        self.beam_size = b_size
        self.checkpoint = checkpoint
        self.decoder = decoder
        self.encoder = encoder
        self.input_folder = Setters()._set_input_folder()
        self.base_data_name = Setters()._set_base_data_name()

    # Load model
    def _load_checkpoint(self):
        print(f"loading checkpoint in {self.checkpoint}")
        if torch.cuda.is_available():

            self.checkpoint = torch.load('../' + self.checkpoint)
        # cpu if not using colab
        else:
            self.checkpoint = torch.load('../' + self.checkpoint, map_location=torch.device('cpu'))

        self.decoder.load_state_dict(self.checkpoint['decoder'])
        self.decoder.to(self.device)
        self.decoder.eval()

        self.encoder.load_state_dict(self.checkpoint['encoder'])
        self.encoder.to(self.device)

        # encoder is frozen
        # encoder.eval()




# import sys
#
# sys.path.insert(0, '/content/gdrive/MyDrive/Tese/code')  # for colab
import os
import torch
from torch import nn
from tqdm import tqdm
from torchvision import transforms
from src.configs.getters.get_data_paths import *
from src.configs.utils.datasets import ClassificationDataset
from src.classification_scripts.SupConLoss.train_supcon import FineTuneSupCon
from src.classification_scripts.set_classification_globals import _set_globals
import sys


continuous = False


class TestSupCon:
    """
    class to test encoder pretrained with SupConLoss
    trains a linear classifier (projection head as the paper suggests)
    """
    def __init__(self):
        self.setters = _set_globals(file  = 'classification_scripts/encoder_training_details.txt')
        logging.info("Device: %s \nCount %i gpus",
                     DEVICE, torch.cuda.device_count())
        self.file = 'classification_scripts/encoder_training_details.txt'

    def _set_transforms(self):
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        return self.normalize

    def _set_loader(self):
        self.val_loader = torch.utils.data.DataLoader(
            ClassificationDataset( self.setters["data_folder"], self.setters["data_name"], 'TEST', continuous=False,
                                  transform=transforms.Compose([self._set_transforms()])),
            batch_size=int(self.setters["h_parameters"]['batch_size']), shuffle=False, num_workers=int(self.setters["h_parameters"]['workers']),
            pin_memory=True)

    def _setup_model(self):
        model = FineTuneSupCon(model_type=ENCODER_MODEL, device=DEVICE, file=self.file)
        self.model = model._setup_train()

    def _load_checkpoint(self):
        if os.path.exists('../' + self.setters["PATHS"]._get_pretrained_encoder_path(encoder_name=ENCODER_LOADER)):
            logging.info("checkpoint exists, loading...")
            if torch.cuda.is_available():
                checkpoint = torch.load('../' + self.setters["PATHS"]._get_pretrained_encoder_path(encoder_name=ENCODER_LOADER))
            else:
                checkpoint = torch.load('../' + self.setters["PATHS"]._get_pretrained_encoder_path(encoder_name=ENCODER_LOADER),
                                        map_location=torch.device("cpu"))
            self.model.load_state_dict(checkpoint['model'])
            logging.info("Model loaded")
    def _setup_eval(self):
        self.model.eval()


logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

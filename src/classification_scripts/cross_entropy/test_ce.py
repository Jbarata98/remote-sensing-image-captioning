# import sys
#
# sys.path.insert(0, '/content/gdrive/MyDrive/Tese/code')  # for colab
import os
import torch
import sys
from torch import nn
from tqdm import tqdm
from torchvision import transforms
from src.configs.getters.get_data_paths import *
from src.configs.utils.datasets import ClassificationDataset
from src.classification_scripts.cross_entropy.train_ce import FineTuneCE
from src.classification_scripts.set_globals import _set_globals


continuous = False



class TestCE:
    """
    class to test encoder pretrained with cross_entropy
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
            ClassificationDataset(self.setters["data_folder"], self.setters["data_name"], 'TEST', continuous=False,
                                  transform=transforms.Compose([self._set_transforms()])),
            batch_size=int(self.setters["h_parameters"]['batch_size']), shuffle=False, num_workers=int(self.setters["h_parameters"]['workers']),
            pin_memory=True)

    def _setup_model(self):
        model = FineTuneCE(model_type=ENCODER_MODEL, device=DEVICE, file = self.file)
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

    def _setup_eval(self):
        self.model.eval()

    def _compute_acc(self):
        # declare dict to initialize
        self.predicted = {}

        # save training details for this experiment
        self.predicted["encoder_training_details"] = self.setters["h_parameters"]

        self.total_acc = torch.tensor([0.0]).to(DEVICE)
        with torch.no_grad():
            for batch, (img, target) in enumerate(tqdm(self.val_loader)):
                if continuous:
                    result = self.model(img)
                    output = torch.sigmoid(result)

                    condition_1 = (output > 0.5)
                    condition_2 = (target == 1)

                    correct_preds = torch.sum(condition_1 * condition_2, dim=1)
                    n_preds = torch.sum(condition_1, dim=1)

                    acc = correct_preds.double() / n_preds
                    acc[torch.isnan(acc)] = 0  # n_preds can be 0
                    acc_batch = torch.mean(acc)

                    self.total_acc += acc_batch

                else:

                    m = nn.Softmax(dim=1)
                    # img = img[0]
                    result = self.model(img.to(DEVICE))
                    output = m(result)
                    # print(output)
                    y = torch.argmax(output.to(DEVICE), dim=1).to(DEVICE)

                    preds = y.detach()

                    targets = target.squeeze(1).to(DEVICE)
                    print(preds, targets)
                    acc_batch = ((preds == targets).float().sum()) / len(preds)

                    self.total_acc += acc_batch

                if batch % 5 == 0:
                    print("acc_batch", acc_batch.item())
                    print("total loss", self.total_acc)

        # print("len of train_data", len(train_loader))
        epoch_acc = (self.total_acc / (batch + 1)).item()
        print("epoch acc", epoch_acc)
        self.predicted["acc_val"] = epoch_acc
        return self.predicted

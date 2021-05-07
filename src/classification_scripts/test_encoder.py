import torch
import os

# import sys
#
# sys.path.insert(0, '/content/gdrive/MyDrive/Tese/code')  # for colab

from src.configs.getters.get_data_paths import *
from src.classification_scripts.train_encoder import PATHS, data_name, data_folder
from src.configs.globals import *
from src.classification_scripts.train_encoder import hparameters, finetune
from src.configs.utils.datasets import ClassificationDataset
from torch import nn
from torchvision import transforms

continuous = False
AUGMENT = True

if __name__ == "__main__":

    logging.basicConfig(
        format='%(levelname)s: %(message)s', level=logging.INFO)
    logging.info("Device: %s \nCount %i gpus",
                 DEVICE, torch.cuda.device_count())

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        ClassificationDataset(data_folder, data_name, 'TRAIN', continuous=False,
                              transform=transforms.Compose([normalize])),
        batch_size=int(hparameters['batch_size']), shuffle=True, num_workers=int(hparameters['workers']),
        pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        ClassificationDataset(data_folder, data_name, 'TEST', continuous=False,
                              transform=transforms.Compose([normalize])),
        batch_size=int(hparameters['batch_size']), shuffle=True, num_workers=int(hparameters['workers']),
        pin_memory=True)

    model = finetune(model_type=ENCODER_MODEL, device=DEVICE)
    model = model._setup_train()

    if os.path.exists('../../' + PATHS._get_checkpoint_path(is_encoder=True, augment=AUGMENT)):
        logging.info("checkpoint exists, loading...")
        if torch.cuda.is_available():
            checkpoint = torch.load('../../' + PATHS._get_checkpoint_path(is_encoder=True, augment=AUGMENT))
        else:
            checkpoint = torch.load('../../' + PATHS._get_checkpoint_path(is_encoder=True, augment=AUGMENT),
                                    map_location=torch.device("cpu"))

    model.load_state_dict(checkpoint['model'])
    model.eval()


    def compute_acc(dataset, train_or_val):
        total_acc = torch.tensor([0.0]).to(DEVICE)

        for batch, (img, target) in enumerate(dataset):
            if continuous:
                result = model(img)
                output = torch.sigmoid(result)

                condition_1 = (output > 0.5)
                condition_2 = (target == 1)

                correct_preds = torch.sum(condition_1 * condition_2, dim=1)
                n_preds = torch.sum(condition_1, dim=1)

                acc = correct_preds.double() / n_preds
                acc[torch.isnan(acc)] = 0  # n_preds can be 0
                acc_batch = torch.mean(acc)

                total_acc += acc_batch

            else:

                m = nn.Softmax(dim=1)
                result = model(img.to(DEVICE))
                output = m(result)
                # print(output)
                y = torch.argmax(output.to(DEVICE), dim=1).to(DEVICE)

                preds = y.detach()

                targets = target.squeeze(1).to(DEVICE)
                print(preds,targets)
                acc_batch = ((preds == targets).float().sum()) / len(preds)

                total_acc += acc_batch

            if batch % 5 == 0:
                print("acc_batch", acc_batch.item())
                print("total loss", total_acc)

        print("len of train_data", len(train_loader))
        epoch_acc = (total_acc / (batch + 1)).item()
        print("epoch acc", train_or_val, epoch_acc)
        return epoch_acc


    # epoch_acc_train = compute_acc(train_loader, "TRAIN")
    epoch_acc_val = compute_acc(val_loader, "TEST")

    # print("train epoch", epoch_acc_train)
    # print("val epoch", epoch_acc_val)

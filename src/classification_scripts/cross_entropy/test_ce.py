# import sys
#
# sys.path.insert(0, '/content/gdrive/MyDrive/Tese/code')  # for colab
import os
import json
import torch
from torch import nn
from tqdm import tqdm
from torchvision import transforms
from src.configs.getters.get_data_paths import *
from src.configs.setters.set_initializers import Setters
from src.configs.utils.datasets import ClassificationDataset
from src.classification_scripts.cross_entropy.train_ce import PATHS, data_name, data_folder, h_parameters, FineTuneCE


continuous = False


def test_CE():
    """
    class to test encoder pretrained with cross_entropy
    """

    logging.basicConfig(
        format='%(levelname)s: %(message)s', level=logging.INFO)
    logging.info("Device: %s \nCount %i gpus",
                 DEVICE, torch.cuda.device_count())

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_loader = torch.utils.data.DataLoader(
        ClassificationDataset(data_folder, data_name, 'TEST', continuous=False,
                              transform=transforms.Compose([normalize])),
        batch_size=int(h_parameters['batch_size']), shuffle=False, num_workers=int(h_parameters['workers']),
        pin_memory=True)

    model = FineTuneCE(model_type=ENCODER_MODEL, device=DEVICE)
    model = model._setup_train()

    if os.path.exists('../../' + PATHS._get_pretrained_encoder_path(encoder_name=ENCODER_LOADER)):
        logging.info("checkpoint exists, loading...")
        if torch.cuda.is_available():
            checkpoint = torch.load('../../' + PATHS._get_pretrained_encoder_path(encoder_name=ENCODER_LOADER))
        else:
            checkpoint = torch.load('../../' + PATHS._get_pretrained_encoder_path(encoder_name=ENCODER_LOADER),
                                    map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # declare dict to initialize
    predicted = {}

    # save training details for this experiment
    predicted["encoder_training_details"] = h_parameters

    def compute_acc_CE(dataset, train_or_val):
        total_acc = torch.tensor([0.0]).to(DEVICE)
        with torch.no_grad():
            for batch, (img, target) in enumerate(tqdm(dataset)):
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
                    # img = img[0]
                    result = model(img.to(DEVICE))
                    output = m(result)
                    # print(output)
                    y = torch.argmax(output.to(DEVICE), dim=1).to(DEVICE)

                    preds = y.detach()

                    targets = target.squeeze(1).to(DEVICE)
                    print(preds, targets)
                    acc_batch = ((preds == targets).float().sum()) / len(preds)

                    total_acc += acc_batch

                if batch % 5 == 0:
                    print("acc_batch", acc_batch.item())
                    print("total loss", total_acc)

        # print("len of train_data", len(train_loader))
        epoch_acc = (total_acc / (batch + 1)).item()
        print("epoch acc", train_or_val, epoch_acc)
        return epoch_acc

    # epoch_acc_train = compute_acc(train_loader, "TRAIN")
    epoch_acc_val = compute_acc_CE(val_loader, "TEST")

    # predicted["acc_train"] = epoch_acc_train
    predicted["acc_val"] = epoch_acc_val

    output_path = '../../' + Setters(file="../encoder_training_details.txt")._set_paths()._get_results_path()

    with open(output_path, 'w+') as f:
        json.dump(predicted, f, indent=2)

test_CE()
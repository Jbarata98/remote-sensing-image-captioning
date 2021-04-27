# import sys
# sys.path.insert(0,'/content/drive/My Drive/Tese/code') #for colab

import os
import torch
import logging
import json
from torch import nn
from torchvision import transforms
import time

from src.configs.get_models import *
from src.configs.globals import *

from src.configs.get_training_optimizers import *
from src.configs.get_training_details import *
from src.configs.datasets import ClassificationDataset
from src.encoder_scripts.create_classification_data import PATHS


FINE_TUNE = True
AUGMENT = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

details = Training_details("encoder_training_details.txt") #name of details file here
hparameters = details._get_training_details()

# set encoder
ENCODER = Encoders(model=ENCODER_MODEL, checkpoint_path='../' + PATHS._load_encoder_path(encoder_loader=ENCODER_LOADER),device = DEVICE)

# set optimizers
OPTIMIZERS = Optimizers(optimizer_type = OPTIMIZER, loss_func=LOSS, device=DEVICE)
DEBUG = False

data_folder = PATHS._get_input_path(is_classification=True)
data_name = DATASET + '_CLASSIFICATION_dataset'

class finetune():

    def __init__(self, model_type, device, nr_classes=31, enable_finetuning= FINE_TUNE):  # default is 31 classes (nr of rscid classes)
        self.device = device
        logging.info("Running encoder fine-tuning script...")

        self.model_type = model_type
        self.classes = nr_classes
        self.enable_finetuning = enable_finetuning
        self.checkpoint_exists = False
        self.device = device

        image_model, dim = ENCODER._get_encoder_model()
        image_model._fc = nn.Linear(dim, self.classes)

        self.model = image_model.to(self.device)

    def _setup_train(self):

        optimizer = OPTIMIZERS._get_optimizer(
            params=filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=float(hparameters['encoder_lr'])) if self.enable_finetuning else None

        self.optimizer = optimizer
        self.criterion = OPTIMIZERS._get_loss_function()
        self._load_weights_from_checkpoint(load_to_train=True)

        return self.model

    def _load_weights_from_checkpoint(self, load_to_train):
        print('../../' + PATHS._get_checkpoint_path(is_encoder=True, augment=AUGMENT))
        if os.path.exists('../../' + PATHS._get_checkpoint_path(is_encoder=True, augment=AUGMENT)):
            logging.info("checkpoint exists, loading...")
            if torch.cuda.is_available():
                checkpoint = torch.load('../../' + PATHS._get_checkpoint_path(is_encoder=True, augment=AUGMENT))
            else:
                checkpoint = torch.load('../../' + PATHS._get_checkpoint_path(is_encoder=True, augment=AUGMENT), map_location=torch.device("cpu"))


            self.checkpoint_exists = True


            # load model weights
            self.model.load_state_dict(checkpoint['model'])

            if load_to_train:
                # load optimizers and start epoch
                self.checkpoint_start_epoch = checkpoint['epoch'] + 1
                self.checkpoint_epochs_since_last_improvement = checkpoint['epochs_since_improvement']
                self.checkpoint_val_loss = checkpoint['val_loss']

                # load weights for encoder

                self.optimizer.load_state_dict(checkpoint['optimizer'])

                logging.info(
                    "Restore model from checkpoint. Start epoch %s ", self.checkpoint_start_epoch)
        else:
            logging.info(
                "No checkpoint. Will start model from beggining\n")

    def _train_step(self, imgs, targets):
        imgs = imgs.to(self.device)
        targets = targets.to(self.device)
        outputs = self.model(imgs)

        targets = targets.squeeze(1)

        loss = self.criterion(outputs, targets)
        self.model.zero_grad()
        loss.backward()

        # Update weights
        self.optimizer.step()

        return loss

    def val_step(self, imgs, targets):
        imgs = imgs.to(self.device)
        targets = targets.to(self.device)
        outputs = self.model(imgs)
        targets = targets.squeeze(1)
        loss = self.criterion(outputs, targets)

        return loss

    def train(self, train_dataloader, val_dataloader, print_freq=int(hparameters['print_freq'])):
        early_stopping = EarlyStopping(
            epochs_limit_without_improvement=6,
            epochs_since_last_improvement=self.checkpoint_epochs_since_last_improvement
            if self.checkpoint_exists else 0,
            baseline=self.checkpoint_val_loss if self.checkpoint_exists else np.Inf,
            encoder_optimizer=self.optimizer,  # TENS
            decoder_optimizer=None,
            period_decay_lr=2  # no decay lr!
        )

        start_epoch = self.checkpoint_start_epoch if self.checkpoint_exists else 0

        # Iterate by epoch
        for epoch in range(start_epoch, int(hparameters['epochs'])):
            self.current_epoch = epoch

            if early_stopping.is_to_stop_training_early():
                break

            start = time.time()
            train_total_loss = 0.0
            val_total_loss = 0.0

            # Train by batch
            self.model.train()

            for batch_i, (imgs, targets) in enumerate(train_dataloader):

                train_loss = self._train_step(
                    imgs, targets
                )

                self._log_status("TRAIN", epoch, batch_i,
                                 train_dataloader, train_loss, print_freq)

                train_total_loss += train_loss

                # (only for debug: interrupt val after 1 step)
                if DEBUG:
                    break

            # End training
            epoch_loss = train_total_loss / (batch_i + 1)
            logging.info('Time taken for 1 epoch {:.4f} sec'.format(
                time.time() - start))
            logging.info('\n\n-----> TRAIN END! Epoch: {}; Loss: {:.4f}\n'.format(epoch,
                                                                                  train_total_loss / (batch_i + 1)))

            # Start validation
            self.model.eval()  # eval mode (no dropout or batchnorm)

            with torch.no_grad():

                for batch_i, (imgs, targets) in enumerate(val_dataloader):

                    val_loss = self.val_step(
                        imgs, targets)

                    self._log_status("VAL", epoch, batch_i,
                                     val_dataloader, val_loss, print_freq)

                    val_total_loss += val_loss

                    # (only for debug: interrupt val after 1 step)
                    if DEBUG:
                        break

            # End validation
            epoch_val_loss = val_total_loss / (batch_i + 1)

            early_stopping.check_improvement(epoch_val_loss)

            self._save_checkpoint_encoder(early_stopping.is_current_val_best(),
                                          epoch,
                                          early_stopping.get_number_of_epochs_without_improvement(),
                                          epoch_val_loss)

            logging.info(
                '\n-------------- END EPOCH:{}⁄{}; Train Loss:{:.4f}; Val Loss:{:.4f} -------------\n'.format(
                    epoch, int(hparameters['epochs']), epoch_loss, epoch_val_loss))

    def _log_status(self, train_or_val, epoch, batch_i, dataloader, loss, print_freq):
        if batch_i % print_freq == 0:
            logging.info(
                "{} - Epoch: [{}/{}]; Batch: [{}/{}]\t Loss: {:.4f}\t".format(
                    train_or_val, epoch, int(hparameters['epochs']), batch_i,
                    len(dataloader), loss
                )
            )

    #Checkpoint saver
    def _save_checkpoint_encoder(self, val_loss_improved, epoch, epochs_since_improvement, val_loss
                                 ):
        if val_loss_improved:
            state = {'epoch': epoch,
                     'epochs_since_improvement': epochs_since_improvement,
                     'val_loss': val_loss,
                     'model': self.model.state_dict(),
                     'optimizer': self.optimizer.state_dict()
                     }

            filename_checkpoint = '../../' + PATHS._get_checkpoint_path(is_encoder = True, augment=AUGMENT)
            torch.save(state, filename_checkpoint)
            # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint


if __name__ == "__main__":
    logging.basicConfig(
        format='%(levelname)s: %(message)s', level=logging.INFO)

    logging.info("Device: %s \nCount %i gpus",
                 DEVICE, torch.cuda.device_count())

    with open(os.path.join(PATHS._get_input_path(is_classification=True),'DICT_LABELS_' + '.json'), 'r') as j:
        classes = json.load(j)

    print("nr of classes:", len(classes))

    #transformation
    data_transform = [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),transforms.RandomRotation(10),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])]
    #loaders
    train_loader = torch.utils.data.DataLoader(
        ClassificationDataset(data_folder, data_name, 'TRAIN', transform=transforms.Compose(data_transform)),
        batch_size=int(hparameters['batch_size']), shuffle=True, num_workers=int(hparameters['workers']), pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        ClassificationDataset(data_folder, data_name, 'VAL', transform=transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])),
        batch_size=int(hparameters['batch_size']), shuffle=False, num_workers=int(hparameters['workers']), pin_memory=True)

    #call functions
    model = finetune(model_type=ENCODER_MODEL, device=DEVICE)
    model._setup_train()
    model.train(train_loader, val_loader)




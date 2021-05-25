import os
import json
from torchvision import transforms
import torch.nn.functional as F

import time
# import sys
#
# sys.path.insert(0, '/content/drive/My Drive/Tese/code')  # for colab
from src.classification_scripts.augment import CustomRotationTransform

from src.configs.utils.datasets import ClassificationDataset
from src.configs.setters.set_initializers import *

FINE_TUNE = True
AUGMENT = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

h_parameters = Setters("encoder_training_details.txt")._set_training_parameters()
PATHS = Setters(file="encoder_training_details.txt")._set_paths()

# set encoder
ENCODER = Setters("encoder_training_details.txt")._set_encoder(pretrained_encoder=ENCODER_LOADER)

# set optimizers
OPTIMIZERS = Setters("encoder_training_details.txt")._set_optimizer()

# set data names
data_folder = PATHS._get_input_path(is_classification=True)
data_name = DATASET + '_CLASSIFICATION_dataset'

DEBUG = False


class FineTune:
    """
    class that unfreezes the efficient-net model and pre-trains it on rsicd data
    """

    def __init__(self, model_type, device, nr_classes=31, enable_finetuning=FINE_TUNE):  # default is 31 classes (nr of rscid classes)
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
            lr=float(h_parameters['encoder_lr'])) if self.enable_finetuning else None

        self.optimizer = optimizer
        self.criterion = OPTIMIZERS._get_loss_function()
        self._load_weights_from_checkpoint(load_to_train=True)

        return self.model

    def _load_weights_from_checkpoint(self, load_to_train):

        print('../../' + PATHS._get_pretrained_encoder_path(encoder_name=ENCODER_LOADER))
        if os.path.exists('../../' + PATHS._get_pretrained_encoder_path(encoder_name=ENCODER_LOADER)):
            logging.info("checkpoint exists, loading...")
            if torch.cuda.is_available():
                checkpoint = torch.load('../../' +PATHS._get_pretrained_encoder_path(encoder_name=ENCODER_LOADER))
            else:
                checkpoint = torch.load('../../' + PATHS._get_pretrained_encoder_path(encoder_name=ENCODER_LOADER),
                                        map_location=torch.device("cpu"))

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
        # if doing diff views on the same batch need to iterate through the list first
        if h_parameters["MULTI_VIEW_BATCH"]:
            img_views =[]
            for i,view in enumerate(imgs):
                img_view = view.to(self.device)
                outputs = self.model(img_view)
                normalized_output = F.normalize(outputs)
                # print(normalized_output.shape)
                img_views.append(normalized_output)
            anchor = img_views[0]
            img_views = torch.transpose(torch.stack(img_views),0,1)

        else:
            img = imgs.to(self.device)
            outputs = self.model(img)
            normalized_output = F.normalize(outputs)

        targets = targets.to(self.device)
        targets = targets.squeeze(1)
        if LOSS == LOSSES.SupConLoss.value:
            loss = self.criterion(normalized_output.unsqueeze(1) if h_parameters["MULTI_VIEW_BATCH"] else img_views, targets)
        else:
            loss = self.criterion(outputs, targets)
        # print(loss)
        top1 = accuracy_encoder(anchor,targets)
        self.model.zero_grad()
        loss.backward()

        # Update weights
        self.optimizer.step()

        return loss,top1,targets.shape[0]

    def val_step(self, imgs, targets):
        if h_parameters["MULTI_VIEW_BATCH"]:
            img_views = []
            for i, view in enumerate(imgs):
                img_view = view.to(self.device)
                outputs = self.model(img_view)
                normalized_output = F.normalize(outputs)
                img_views.append(normalized_output)
            anchor = img_views[0]
            img_views = torch.transpose(torch.stack(img_views), 0, 1)

        else:
            img = imgs.to(self.device)
            outputs = self.model(img)
            normalized_output = F.normalize(outputs)

        targets = targets.to(self.device)
        targets = targets.squeeze(1)
        if LOSS == LOSSES.SupConLoss.value:
            loss = self.criterion(normalized_output.unsqueeze(1) if h_parameters["MULTI_VIEW_BATCH"] else img_views,
                                  targets)
        else:
            loss = self.criterion(outputs, targets)
        top1 = accuracy_encoder(anchor,targets)

        return loss, top1, targets.shape[0]

    def train(self, train_dataloader, val_dataloader, print_freq=int(h_parameters['print_freq'])):
        early_stopping = EarlyStopping(
            epochs_limit_without_improvement=6,
            epochs_since_last_improvement=self.checkpoint_epochs_since_last_improvement
            if self.checkpoint_exists else 0,
            baseline=self.checkpoint_val_loss if self.checkpoint_exists else np.Inf,
            encoder_optimizer=self.optimizer,  # TENS
            decoder_optimizer=None,
            period_decay_lr=2  # no decay lr!
        )
        batch_time = AverageMeter()
        train_losses = AverageMeter()
        val_losses = AverageMeter()
        train_top1accs = AverageMeter()
        val_top1accs = AverageMeter()

        start = time.time()

        start_epoch = self.checkpoint_start_epoch if self.checkpoint_exists else 0

        # Iterate by epoch
        for epoch in range(start_epoch, int(h_parameters['epochs'])):
            self.current_epoch = epoch

            if early_stopping.is_to_stop_training_early():
                break


            # Train by batch
            self.model.train()

            for batch_i, (imgs, targets) in enumerate(train_dataloader):

                train_loss,top1, bsz = self._train_step(
                    imgs, targets
                )

                train_losses.update(train_loss.item(),bsz)
                train_top1accs.update(top1[0].item(),bsz)
                self._log_status("TRAIN", epoch, batch_i,
                                 train_dataloader, train_loss,top1[0].item(), print_freq)


                # (only for debug: interrupt val after 1 step)
                if DEBUG:
                    break
                batch_time.update(time.time() - start)
            # End training
            logging.info('Average time taken for batch {:.4f} sec'.format(
               batch_time))
            logging.info('\n\n-----> TRAIN END! Epoch: {}; Loss: {:.4f}; Top1-Accuracy: {:.4f};\n'.format(epoch,
                                                                                  train_losses,train_top1accs))

            # Start validation
            self.model.eval()  # eval mode (no dropout or batchnorm)

            with torch.no_grad():

                for batch_i, (imgs, targets) in enumerate(val_dataloader):

                    val_loss, top1, bsz = self.val_step(
                        imgs, targets)
                    val_losses.update(val_loss.item(), bsz)
                    val_top1accs.update(top1[0].item(), bsz)
                    self._log_status("VAL", epoch, batch_i,
                                     val_dataloader, val_loss,top1[0].item(), print_freq)



                    # (only for debug: interrupt val after 1 step)
                    if DEBUG:
                        break

            # End validation

            early_stopping.check_improvement(val_losses)

            self._save_checkpoint_encoder(early_stopping.is_current_val_best(),
                                          epoch,
                                          early_stopping.get_number_of_epochs_without_improvement(),
                                          val_losses)

            logging.info(
                '\n-------------- END EPOCH:{}⁄{}; Train Loss:{:.4f}; Val Loss:{:.4f};Train Acc:{:.4f}; Val Acc:{:.4f} -------------\n'.format(
                    epoch, int(h_parameters['epochs']), train_losses, val_losses, train_top1accs, val_top1accs))

    def _log_status(self, train_or_val, epoch, batch_i, dataloader, loss, acc, print_freq):
        if batch_i % print_freq == 0:
            logging.info(
                "{} - Epoch: [{}/{}]; Batch: [{}/{}]\t Loss: {:.4f}\t Acc: {:.4f}\t".format(
                    train_or_val, epoch, int(h_parameters['epochs']), batch_i,
                    len(dataloader), loss,acc
                )
            )

    # Checkpoint saver
    def _save_checkpoint_encoder(self, val_loss_improved, epoch, epochs_since_improvement, val_loss
                                 ):
        if val_loss_improved:
            state = {'epoch': epoch,
                     'epochs_since_improvement': epochs_since_improvement,
                     'val_loss': val_loss,
                     'model': self.model.state_dict(),
                     'optimizer': self.optimizer.state_dict()
                     }

            filename_checkpoint = '../../' + PATHS._get_pretrained_encoder_path(encoder_name=ENCODER_LOADER)
            torch.save(state, filename_checkpoint)
            # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint


if __name__ == "__main__":
    logging.basicConfig(
        format='%(levelname)s: %(message)s', level=logging.INFO)

    logging.info("Device: %s \nCount %i gpus",
                 DEVICE, torch.cuda.device_count())

    with open(os.path.join(PATHS._get_input_path(is_classification=True), 'DICT_LABELS_' + '.json'), 'r') as j:
        classes = json.load(j)

    print("nr of classes:", len(classes))

    # transformation
    data_transform = [transforms.RandomHorizontalFlip(),
                      transforms.RandomVerticalFlip(),
                      CustomRotationTransform(angles=[90, 180, 270]),
                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])]

    # loaders
    train_loader = torch.utils.data.DataLoader(
        ClassificationDataset(data_folder, data_name, 'TRAIN', transform=transforms.Compose(data_transform)),
        batch_size=int(h_parameters['batch_size']), shuffle=True, num_workers=int(h_parameters['workers']),
        pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        ClassificationDataset(data_folder, data_name, 'VAL',
                              transform=transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                 std=[0.229, 0.224, 0.225])])),
        batch_size=int(h_parameters['batch_size']), shuffle=False, num_workers=int(h_parameters['workers']),
        pin_memory=True)

    # call functions
    model = FineTune(model_type=ENCODER_MODEL, device=DEVICE)
    model._setup_train()
    model.train(train_loader, val_loader)

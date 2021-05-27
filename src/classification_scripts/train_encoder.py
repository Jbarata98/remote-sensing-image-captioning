import json

import h5py
from torchvision import transforms
import torch.nn.functional as F

import time
# import sys
#
# sys.path.insert(0,'/content/drive/My Drive/Tese/code')  # for colab
from src.classification_scripts.augment import CustomRotationTransform
from src.classification_scripts.augment import TwoViewTransform

from src.configs.utils.datasets import ClassificationDataset
from src.configs.setters.set_initializers import *

FINE_TUNE = True

setters_class = Setters(file ="encoder_training_details.txt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# training parameters
h_parameters = setters_class._set_training_parameters()
PATHS = setters_class._set_paths()

# set encoder
ENCODER = setters_class._set_encoder(
    path='../' + PATHS._get_pretrained_encoder_path(
        encoder_name=ENCODER_LOADER))

# set optimizers
OPTIMIZERS = setters_class._set_optimizer()

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
        self.device = device
        self.checkpoint_exists = False

        image_model, dim = ENCODER._get_encoder_model()

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

        if os.path.exists('../../' + PATHS._get_pretrained_encoder_path(encoder_name=ENCODER_LOADER)):
            logging.info("checkpoint exists, loading...")
            if torch.cuda.is_available():
                checkpoint = torch.load('../../' + PATHS._get_pretrained_encoder_path(encoder_name=ENCODER_LOADER))
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

        # print(h_parameters["MULTI_VIEW_BATCH"])
        if h_parameters["MULTI_VIEW_BATCH"] == 'True':
            img_views = []
            for i, view in enumerate(imgs):
                img_view = view.to(self.device)
                print(img_view.shape)
                outputs = self.model(img_view)
                if i == 0:
                    anchor = outputs
                normalized_output = F.normalize(outputs)
                # print(normalized_output.shape)
                img_views.append(normalized_output)

            img_views = torch.transpose(torch.stack(img_views), 0, 1)

        else:
            img = imgs.to(self.device)
            outputs = self.model(img)
            normalized_output = F.normalize(outputs)

        targets = targets.to(self.device)
        targets = targets.squeeze(1)
        if LOSS == LOSSES.SupConLoss.value:
            if h_parameters["MULTI_VIEW_BATCH"] == 'True':
                loss = self.criterion(img_views, targets)
            else:
                loss = self.criterion(normalized_output.unsqueeze(1), targets)
            top5 = accuracy_encoder(anchor, targets, topk=(5,))

        else:
            loss = self.criterion(outputs, targets)
            top5 = accuracy_encoder(outputs, targets, topk=(5,))

        # print(loss)
        self.model.zero_grad()
        loss.backward()

        # Update weights
        self.optimizer.step()

        return loss, top5, targets.shape[0]

    def val_step(self, imgs, targets):

        if h_parameters["MULTI_VIEW_BATCH"] == 'True':
            img_views = []
            for i, view in enumerate(imgs):
                img_view = view.to(self.device)
                outputs = self.model(img_view)

                if i == 0:
                    anchor_val = outputs

                normalized_output = F.normalize(outputs)
                img_views.append(normalized_output)
            img_views = torch.transpose(torch.stack(img_views), 0, 1)

        else:
            img = imgs.to(self.device)
            outputs = self.model(img)
            normalized_output = F.normalize(outputs)

        targets = targets.to(self.device)
        targets = targets.squeeze(1)
        if LOSS == LOSSES.SupConLoss.value:
            loss = self.criterion(img_views if h_parameters["MULTI_VIEW_BATCH"] == 'True' else normalized_output.unsqueeze(1),
                                  targets)
            top5 = accuracy_encoder(anchor_val, targets)

        else:
            loss = self.criterion(outputs, targets)
            top5 = accuracy_encoder(outputs, targets, topk=(5,))


        return loss, top5, targets.shape[0]

    def train(self, train_dataloader, val_dataloader, print_freq=int(h_parameters['print_freq'])):
        early_stopping = EarlyStopping(
            epochs_limit_without_improvement=6,
            epochs_since_last_improvement=self.checkpoint_epochs_since_last_improvement
            if self.checkpoint_exists else 0,
            baseline=torch.FloatTensor([self.checkpoint_val_loss.val]) if self.checkpoint_exists else np.Inf,
            encoder_optimizer=self.optimizer,  # TENS
            decoder_optimizer=None,
            period_decay_lr=2  # no decay lr!
        )
        batch_time = AverageMeter()
        train_losses = AverageMeter()
        val_losses = AverageMeter()
        train_top5accs = AverageMeter()
        val_top5accs = AverageMeter()

        start = time.time()
        #
        start_epoch = self.checkpoint_start_epoch if self.checkpoint_exists else 0
        #
        # Iterate by epoch
        for epoch in range(start_epoch, int(h_parameters['epochs'])):
            self.current_epoch = epoch

            if early_stopping.is_to_stop_training_early():
                break

            # Train by batch
            self.model.train()

            for batch_i, (imgs, targets) in enumerate(train_dataloader):

                train_loss, top5, bsz = self._train_step(
                    imgs, targets
                )

                train_losses.update(train_loss.item(), bsz)
                train_top5accs.update(top5[0].item(), bsz)
                self._log_status("TRAIN", epoch, batch_i,
                                 train_dataloader, train_loss, top5[0].item(), print_freq)

                # (only for debug: interrupt val after 1 step)
                if DEBUG:
                    break
                batch_time.update(time.time() - start)
            # End training
            logging.info(' Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t sec'.format(
                batch_time=batch_time))
            logging.info('\n\n-----> TRAIN END! Epoch: {}\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(epoch,
                                                                                   loss=train_losses,
                                                                                   top5=train_top5accs))

            # Start validation
            self.model.eval()  # eval mode (no dropout or batchnorm)

            with torch.no_grad():

                for batch_i, (imgs, targets) in enumerate(val_dataloader):

                    val_loss, top5, bsz = self.val_step(
                        imgs, targets)
                    val_losses.update(val_loss.item(), bsz)
                    val_top5accs.update(top5[0].item(), bsz)
                    self._log_status("VAL", epoch, batch_i,
                                     val_dataloader, val_loss, top5[0].item(), print_freq)

                    # (only for debug: interrupt val after 1 step)
                    if DEBUG:
                        break

            # End validation

            early_stopping.check_improvement(torch.Tensor([val_losses.avg]))

            self._save_checkpoint_encoder(early_stopping.is_current_val_best(),
                                          epoch,
                                          early_stopping.get_number_of_epochs_without_improvement(),
                                          val_losses)

            logging.info(
                '\n-------------- END EPOCH:{}⁄{}\t  Train Loss {train_loss.val:.4f} ({train_loss.avg:.4f})\t'
                'Val Loss {val_loss.val:.4f} ({val_loss.avg:.4f})\t'
                'top-5 Train Accuracy {top5_train.val:.3f} ({top5_train.avg:.3f})\t'
                'top-5 Val Accuracy {top5_val.val:.3f} ({top5_val.avg:.3f})\t'
                    .format(
                    epoch, int(h_parameters['epochs']), train_loss=train_losses, val_loss=val_losses,
                    top5_train=train_top5accs, top5_val=val_top5accs))

    def _log_status(self, train_or_val, epoch, batch_i, dataloader, loss, acc, print_freq):
        if batch_i % print_freq == 0:
            logging.info(
                "{} - Epoch: [{}/{}]; Batch: [{}/{}]\t Loss: {:.4f}\t Acc: {:.4f}\t".format(
                    train_or_val, epoch, int(h_parameters['epochs']), batch_i,
                    len(dataloader), loss, acc
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
    print("h_parameters:", h_parameters)
    print("nr of classes:", len(classes))

    # load target images for histogram matching if dealing with training data
    target_h = h5py.File(os.path.join(data_folder, 'TEST_IMAGES_' + data_name + '.hdf5'), 'r')
    target_imgs = target_h['images']

    data_transform = [transforms.RandomHorizontalFlip(),
                      transforms.RandomVerticalFlip(),
                      CustomRotationTransform(angles=[90, 180, 270]),
                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])]

    # loaders
    train_loader = torch.utils.data.DataLoader(
        ClassificationDataset(data_folder, data_name, 'TRAIN', transform=TwoViewTransform(transforms.Compose(data_transform), target_imgs)),
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

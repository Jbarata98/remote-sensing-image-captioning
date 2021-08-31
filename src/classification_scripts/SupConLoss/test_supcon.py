# import sys
#
# sys.path.insert(0, '/content/gdrive/MyDrive/Tese/code')  # for colab
import os
import time

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torchvision import transforms

from src.classification_scripts.augment import TwoViewTransform, CustomRotationTransform
from src.configs.getters.get_data_paths import *
from src.configs.getters.get_training_optimizers import AverageMeter, accuracy_encoder, EarlyStopping
from src.configs.utils.datasets import ClassificationDataset
from src.classification_scripts.SupConLoss.SupConModel import SupConEffNet, LinearClassifier
from src.classification_scripts.set_classification_globals import _set_globals
import sys

continuous = False


class TestSupCon:
    """
    class to test encoder pretrained with SupConLoss
    trains a linear classifier (projection head as the paper suggests)
    """

    def __init__(self):
        self.setters = _set_globals(file='classification_scripts/encoder_training_details.txt')
        logging.info("Device: %s \nCount %i gpus",
                     DEVICE, torch.cuda.device_count())
        self.file = 'classification_scripts/encoder_training_details.txt'

    def _set_transforms(self):
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        return self.normalize

    def _setup_dataloaders(self):
        self.data_transform = [transforms.RandomHorizontalFlip(),
                               transforms.RandomVerticalFlip(),
                               CustomRotationTransform(angles=[90, 180, 270]),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])]
        # loaders
        self.train_loader = torch.utils.data.DataLoader(
            ClassificationDataset(self.setters["data_folder"], self.setters["data_name"],'TRAIN',
                                  transform=transforms.Compose(self.data_transform)),
            batch_size=int(self.setters["h_parameters"]['batch_size']), shuffle=True,
            num_workers=int(self.setters["h_parameters"]['workers']),
            pin_memory=True)

        self.val_loader = torch.utils.data.DataLoader(
            ClassificationDataset(self.setters["data_folder"], self.setters["data_name"], 'VAL',
                                  transform=transforms.Compose(
                                      [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])),
            batch_size=int(self.setters["h_parameters"]['batch_size']), shuffle=False,
            num_workers=int(self.setters["h_parameters"]['workers']),
            pin_memory=True)
        return self.train_loader, self.val_loader

    def _setup_model(self, eff_net_version = 'v1'):
        self.eff_net_version = eff_net_version
        print(eff_net_version)
        self.model = SupConEffNet(eff_net_version= self.eff_net_version)
        # use cross entropy for training the linear classifier
        self.criterion = torch.nn.CrossEntropyLoss()

        # nr class default is 31
        self.classifier = LinearClassifier(eff_net_version=self.eff_net_version)

        self.state_dict = self._load_checkpoint()

        new_state_dict = {}
        for k, v in self.state_dict.items():
            k = k.replace("module.", "")
            new_state_dict[k] = v
            state_dict = new_state_dict
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.classifier =self. classifier.cuda()
            self.criterion = self.criterion.cuda()
            cudnn.benchmark = True

        self.model.load_state_dict(self.state_dict)
        logging.info("Model loaded")

        return self.model, self.classifier, self.criterion

    def _load_checkpoint(self):
        if os.path.exists('../' + self.setters["PATHS"]._get_pretrained_encoder_path(encoder_name=ENCODER_LOADER)):
            logging.info("checkpoint exists, loading...")
            if torch.cuda.is_available():
                checkpoint = torch.load(
                    '../' + self.setters["PATHS"]._get_pretrained_encoder_path(encoder_name=ENCODER_LOADER))
            else:
                checkpoint = torch.load(
                    '../' + self.setters["PATHS"]._get_pretrained_encoder_path(encoder_name=ENCODER_LOADER),
                    map_location=torch.device("cpu"))
            state_dict = checkpoint["model"]


            return state_dict

    def _setup_eval(self):
        self.model.eval()

    def _train_classifier(self, train_loader, model, classifier, criterion, optimizer, epoch):
        """one epoch training"""
        model.eval()
        classifier.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

        end = time.time()
        for idx, (images, labels) in enumerate(train_loader):
            data_time.update(time.time() - end)

            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            bsz = labels.shape[0]

            # compute loss
            with torch.no_grad():
                if self.eff_net_version == 'v1':
                    features = model.model.extract_features(images)
                elif self.eff_net_version =='v2':
                    features = model.model.forward_features(images)


            output = classifier(features.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2).mean(dim=1).detach())
            # print(labels.shape)
            # print(labels)

            loss = criterion(output, labels.squeeze(1))

            # update metric
            losses.update(loss.item(), bsz)

            acc1, acc5 = accuracy_encoder(output, labels.squeeze(1), topk=(1,5))
            top1.update(acc1[0], bsz)

            # SGD
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print info
            if (idx) % int(self.setters["h_parameters"]["print_freq"]) == 0:
                print('Train: [{0}][{1}/{2}]\t'
                      'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    epoch, idx + 1, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1))
                sys.stdout.flush()

        return losses.avg, top1.avg

    def validate_classifier(self, val_loader, model, classifier, criterion):
        """validation"""
        model.eval()
        classifier.eval()

        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

        with torch.no_grad():
            end = time.time()
            for idx, (images, labels) in enumerate(val_loader):
                if torch.cuda.is_available():
                    images = images.float().cuda()
                    labels = labels.cuda()
                bsz = labels.shape[0]

                # forward
                if self.eff_net_version == 'v1':
                    output = classifier(model.model.extract_features(images).permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2).mean(dim=1))
                elif self.eff_net_version =='v2':
                    output = classifier(model.model.forward_features(images).permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2).mean(dim=1))

                loss = criterion(output, labels.squeeze(1))

                # update metric
                losses.update(loss.item(), bsz)
                acc1, acc5 = accuracy_encoder(output, labels, topk=(1, 5))
                top1.update(acc1[0], bsz)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if idx % int(self.setters["h_parameters"]["print_freq"]) == 0:
                    print('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                        idx, len(val_loader), batch_time=batch_time,
                        loss=losses, top1=top1))

        print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
        return losses.avg, top1.avg

    def _train(self, eff_net_version = 'v1'):
        self.eff_net_version = eff_net_version
        best_acc = 0

        # build data loader
        train_loader, val_loader = self._setup_dataloaders()

        # build model and criterion
        model, classifier, criterion = self._setup_model(eff_net_version=self.eff_net_version)

        optimizer = self.setters["OPTIMIZERS"]._get_optimizer(
            params=filter(lambda p: p.requires_grad, self.classifier.parameters()),
            lr=float(self.setters["h_parameters"]['encoder_lr']))

        early_stopping = EarlyStopping(
            epochs_limit_without_improvement=6,
            epochs_since_last_improvement=0,
            baseline=np.Inf,
            encoder_optimizer=optimizer,  # TENS
            decoder_optimizer=None,
            period_decay_lr=2  # no decay lr!
        )
        # training routine
        logging.info("Train Started...")
        for epoch in range(1, int(self.setters["h_parameters"]["epochs"]) + 1):
            if early_stopping.is_to_stop_training_early():
                break
            # train for one epoch
            time1 = time.time()
            loss, acc = self._train_classifier(train_loader, model, classifier, criterion,
                                               optimizer, epoch)
            time2 = time.time()
            print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(
                epoch, time2 - time1, acc))

            # eval for one epoch
            val_loss, val_acc = self.validate_classifier(val_loader, model, classifier, criterion)
            if val_acc > best_acc:
                best_acc = val_acc

            early_stopping.check_improvement(torch.Tensor([val_loss]))

        print('best accuracy: {:.2f}'.format(best_acc))


logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

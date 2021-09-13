# import sys
#
# sys.path.insert(0, '/content/gdrive/MyDrive/Tese/code')  # for colab
import time
import torch.nn.functional as F
from src.classification_scripts.finetune_abstract import *


class FineTuneSupCon(FineTune):
    """
    class that unfreezes the efficient-net model and pre-trains it on RSICD data
    """

    def __init__(self, model_type, device, file, nr_classes=31, eff_net_version = 'v1'):  # default is 31 classes (nr of rscid classes)

        super().__init__(model_type, device, file, nr_classes, eff_net_version)


    def _train_step(self, imgs, targets):

        # if doing diff views on the same batch need to iterate through the list first


        images = torch.cat([imgs[0], imgs[1]], dim = 0)
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
        bsz = targets.shape[0]

        features = self.model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        # print(features.shape, targets.squeeze(1).shape)
        loss = self.criterion(features, targets.squeeze(1))


        self.model.zero_grad()
        loss.backward()

        # Update weights
        self.optimizer.step()

        return loss, targets.shape[0]

    def val_step(self, imgs, targets):

        """
        validation step
        """
        # if doing diff views on the same batch need to iterate through the list first

        images = torch.cat([imgs[0], imgs[1]], dim=0)
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
        bsz = targets.shape[0]

        features = self.model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        loss = self.criterion(features, targets.squeeze(1))

        return loss, targets.shape[0]

    def train(self, train_dataloader, val_dataloader):
        """
        train the model
        """
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

        start = time.time()
        #
        start_epoch = self.checkpoint_start_epoch if self.checkpoint_exists else 0
        #
        # Iterate by epoch
        for epoch in range(start_epoch, int(self.setters["h_parameters"]['epochs'])):
            self.current_epoch = epoch

            if early_stopping.is_to_stop_training_early():
                break

            # #Train by batch
            self.model.train()

            for batch_i, (imgs, targets) in enumerate(train_dataloader):

                train_loss, bsz = self._train_step(imgs, targets)

                train_losses.update(train_loss.item(), bsz)
                self._log_status("TRAIN", epoch, batch_i,
                                 train_dataloader, train_loss)

                # (only for debug: interrupt val after 1 step)
                if self.setters["DEBUG"]:
                    break
                batch_time.update(time.time() - start)
            # End training
            logging.info(' Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t sec'.format(
                batch_time=batch_time))
            logging.info('\n\n-----> TRAIN END! Epoch: {}\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, loss=train_losses))

            # # Start validation
            self.model.eval()  # eval mode (no dropout or batchnorm)

            with torch.no_grad():

                for batch_i, (imgs, targets) in enumerate(val_dataloader):

                    val_loss, bsz = self.val_step(imgs, targets)
                    val_losses.update(val_loss.item(), bsz)
                    self._log_status("VAL", epoch, batch_i,
                                     val_dataloader, val_loss)

                    # (only for debug: interrupt val after 1 step)
                    if self.setters["DEBUG"]:
                        break

            # End validation

            early_stopping.check_improvement(torch.Tensor([val_losses.avg]))

            self._save_checkpoint_encoder(early_stopping.is_current_val_best(),
                                          epoch,
                                          early_stopping.get_number_of_epochs_without_improvement(),
                                          val_losses)

            logging.info(
                '\n-------------- END EPOCH:{}⁄{}\t  Train Loss {train_loss.val:.4f} ({train_loss.avg:.4f})\t'
                'Val Loss {val_loss.val:.4f} ({val_loss.avg:.4f})\t'.format(
                    epoch, int(self.setters["h_parameters"]['epochs']), train_loss=train_losses, val_loss=val_losses))

    def _log_status(self, train_or_val, epoch, batch_i, dataloader, loss):
        print_freq = int(self.setters["h_parameters"]['print_freq'])
        if batch_i % print_freq == 0:
            logging.info(
                "{} - Epoch: [{}/{}]; Batch: [{}/{}]\t Loss: {:.4f}\t".format(
                    train_or_val, epoch, int(self.setters["h_parameters"]['epochs']), batch_i,
                    len(dataloader), loss
                )
            )

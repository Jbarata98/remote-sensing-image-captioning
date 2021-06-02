# import sys
#
# sys.path.insert(0, '/content/gdrive/MyDrive/Tese/code')  # for colab

from src.classification_scripts.finetune_abstract import *


class FineTuneCE(FineTune):
    """
    class that unfreezes the efficient-net model and pre-trains it on RSICD data
    """

    def __init__(self, model_type, device, nr_classes=31,
                 enable_finetuning=FINE_TUNE):  # default is 31 classes (nr of rscid classes)

        super().__init__(model_type, device, nr_classes, enable_finetuning)

    def _train_step(self, imgs, targets):

        img = imgs.to(self.device)
        outputs = self.model(img)

        targets = targets.to(self.device)
        targets = targets.squeeze(1)

        # using cross-entropy

        loss = self.criterion(outputs, targets)
        # test accuracy when running cross_entropy
        top5 = accuracy_encoder(outputs, targets, topk=(5,))

        # print(loss)
        self.model.zero_grad()
        loss.backward()

        # Update weights
        self.optimizer.step()

        return loss, top5, targets.shape[0]

    def val_step(self, imgs, targets):
        """
        validation step
        """

        img = imgs.to(self.device)
        outputs = self.model(img)

        targets = targets.to(self.device)
        targets = targets.squeeze(1)

        loss = self.criterion(outputs, targets)
        top5 = accuracy_encoder(outputs, targets, topk=(5,))

        return loss, top5, targets.shape[0]

    def train(self, train_dataloader, val_dataloader, print_freq=int(h_parameters['print_freq'])):
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





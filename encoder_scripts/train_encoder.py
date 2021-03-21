from configs.utils import *
from encoder_scripts.encoder_training_details import *
from encoder_scripts.create_classification_data import create_classes_json,create_classification_files

import os


class finetune():

    def __init__(self, model_type, device, nr_classes=31, enable_finetuning=True):  # rsicd is 31 classes
        self.device = device
        logging.info("Running encoder fine-tuning script...")

        self.model_type = model_type
        self.classes = nr_classes
        self.enable_finetuning = enable_finetuning
        self.checkpoint_exists = False
        self.device = device

        image_model, dim = get_encoder_model(self.model_type)
        image_model._fc = nn.Linear(dim, self.classes)

        self.model = image_model.to(self.device)

    def _setup_train(self):

        optimizer = get_optimizer(OPTIMIZER)(
            params=filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=encoder_lr) if self.enable_finetuning else None

        self.optimizer = optimizer

        self.criterion = get_loss_function(LOSS)


        self._load_weights_from_checkpoint(load_to_train=True)

        return self.model

    def _load_weights_from_checkpoint(self, load_to_train):

        if os.path.exists(checkpoint_encoder_path):
            logging.info("checkpoint exists, loading...")
            checkpoint = torch.load(checkpoint_encoder_path, map_location=torch.device("cpu"))
            self.checkpoint_exists = True

            checkpoint = torch.load(checkpoint_encoder_path,  map_location=torch.device("cpu"))

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

    def train(self, train_dataloader, val_dataloader, print_freq=print_freq):
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
        for epoch in range(start_epoch, epochs):
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
                    epoch, epochs, epoch_loss, epoch_val_loss))

    def _log_status(self, train_or_val, epoch, batch_i, dataloader, loss, print_freq):
        if batch_i % print_freq == 0:
            logging.info(
                "{} - Epoch: [{}/{}]; Batch: [{}/{}]\t Loss: {:.4f}\t".format(
                    train_or_val, epoch, epochs, batch_i,
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

            filename_checkpoint = get_path(model=ENCODER_MODEL, is_encoder=True)
            torch.save(state, filename_checkpoint)
            # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint


if __name__ == "__main__":
    logging.basicConfig(
        format='%(levelname)s: %(message)s', level=logging.INFO)

    logging.info("Device: %s \nCount %i gpus",
                 device, torch.cuda.device_count())

    #create a json with the classes, basically a classification dataset
    create_classes_json()
    #create the files (images and labels splits)
    NR_CLASSES = create_classification_files(DATASET, get_classification_dataset_path(DATASET),
                                             get_images_path(DATASET),
                                             get_path(classification=True, input=True))

    #transformation
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    #loaders
    train_loader = torch.utils.data.DataLoader(
        ClassificationDataset(data_folder, data_name, 'TRAIN', transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        ClassificationDataset(data_folder, data_name, 'VAL', transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    #call functions
    model = finetune(model_type=ENCODER_MODEL, device=device)
    model._setup_train()
    model.train(train_loader, val_loader)




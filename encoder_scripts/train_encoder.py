from models.base_model import Encoder
from configs.utils import *
from encoder_scripts.encoder_training_details import *
import sys
import os

class finetune():

    def __init__(self, model_type, nr_classes, enable_finetuning):
        logging.info("Running encoder fine-tuning script...")

        self.model_type = model_type
        self.classes = nr_classes
        self.enable_finetuning = enable_finetuning

        if os.path.exists(checkpoint_encoder_path):
            logging.info("checkpoint exists, loading...")
            checkpoint = torch.load(checkpoint_encoder_path)

            self.start_epoch = checkpoint['epoch'] + 1
            self.epochs_since_improvement = checkpoint['epochs_since_improvement']
            self.val_loss = checkpoint['val_loss']

            #load weights for encoder
            encoder = checkpoint['model']
            optimizer = checkpoint_encoder_path['optimizer']

        else:
            logging.info("checkpoint does not exist... starting from the start")
            encoder = get_encoder_model(self.model_type)
            num_features = encoder._fc.in_features
            encoder._fc = nn.Linear(num_features, self.classes)

            optimizer = get_optimizer(OPTIMIZER)(
                params=filter(lambda p: p.requires_grad, encoder.parameters()),
                lr=encoder_lr) if enable_finetuning else None

        self.encoder_model = encoder.to(device)
        self.encoder_optimizer = optimizer
        self.criterion = get_loss_function(LOSS)


        def train_step(self, imgs, targets):
            imgs = imgs.to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(imgs)

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
            loss = self.criterion(outputs, targets)

            return loss

        def train(self, train_dataloader, val_dataloader, print_freq=print_freq):
            early_stopping = EarlyStopping(
                epochs_limit_without_improvement=6,
                epochs_since_last_improvement=self.checkpoint_epochs_since_last_improvement
                if self.checkpoint_exists else 0,
                baseline=self.checkpoint_val_loss if self.checkpoint_exists else np.Inf,
                encoder_optimizer= self.encoder_optimizer,  # TENS
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

                    train_loss = self.train_step(
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

                self._save_checkpoint(early_stopping.is_current_val_best(),
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

        def save_checkpoint_encoder(self, epoch, epochs_since_improvement, val_loss,
                                    is_best):

            state = {'epoch': epoch,
                     'epochs_since_last_improvement': epochs_since_improvement,
                     'val_loss': val_loss,
                     'model': self.encoder_model.state_dict(),
                     'optimizer': self.encoder_optimizer.state_dict()
                     }

            filename_checkpoint = get_path(model=ENCODER_MODEL, best_checkpoint=False, is_encoder=True, fine_tune=True)
            torch.save(state, filename_checkpoint)
            # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
            if is_best:
                filename_best_checkpoint = get_path(model=ENCODER_MODEL, best_checkpoint=True, is_encoder=True,
                                                    fine_tune=True)
                torch.save(state, filename_best_checkpoint)


    if __name__ == "__main__":

        logging.basicConfig(
            format='%(levelname)s: %(message)s', level=logging.INFO)


        logging.info("Device: %s \nCount %i gpus",
                     device, torch.cuda.device_count())

        # if DATASET == DATASETS.RSICD.value:
        #     CLASSIFICATION_DATASET_PATH = "classification_dataset"
        # elif DATASET == DATASETS.UCM.value:
        #     CLASSIFICATION_DATASET_PATH = "classification_dataset_ucm"
        # elif DATASET == DATASETS.SYDNEY.value:
        #     CLASSIFICATION_DATASET_PATH = "classification_dataset_flickr8k"
        # else:
        #     raise Exception("Invalid dataset")
        #
        # print("Path of classificaion dataset", DATASET)
        #
        # dataset_folder, dataset_jsons = get_dataset_paths(DATASET)
        # print("dataset folder", dataset_folder)
        #
        # classification_state = torch.load(dataset_jsons + CLASSIFICATION_DATASET_PATH)
        # classes_to_id = classification_state["classes_to_id"]
        # id_to_classes = classification_state["id_to_classes"]
        # classification_dataset = classification_state["classification_dataset"]
        #
        # dataset_len = len(classification_dataset)
        # split_ratio = int(dataset_len * 0.10)
        #
        # classification_train = dict(list(classification_dataset.items())[split_ratio:])
        # classification_val = dict(list(classification_dataset.items())[0:split_ratio])
        #
        # train_dataset_args = (classification_train, dataset_folder + "raw_dataset/images/", classes_to_id)
        # val_dataset_args = (classification_val, dataset_folder + "raw_dataset/images/", classes_to_id)
        #
        # train_dataloader = DataLoader(
        #     ClassificationDataset(*train_dataset_args),
        #     batch_size=BATCH_SIZE,
        #     shuffle=True,
        #     num_workers=NUM_WORKERS
        # )
        #
        # val_dataloader = DataLoader(
        #     ClassificationDataset(*val_dataset_args),
        #     batch_size=BATCH_SIZE,
        #     shuffle=False,
        #     num_workers=NUM_WORKERS
        # )
        #
        # vocab_size = len(classes_to_id)
        #
        # model = ClassificationModel(vocab_size, device)
        # model.setup_to_train()
        # model.train(train_dataloader, val_dataloader)




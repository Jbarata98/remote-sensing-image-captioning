import os

from torchvision import transforms

#
# import sys
#
# sys.path.append('/content/gdrive/MyDrive/Tese/code')
from src.configs.utils.datasets import CaptionDataset
from src.configs.setters.set_initializers import *



# training details on file configs/setters/training_details.txt


class AbstractTrain:
    """
    Training and validation of the model.
    """

    def __init__(self, language_aux, fine_tune_encoder=False, word_map=None, device=DEVICE):

        self.training_parameters = Setters()._set_training_parameters()
        self.checkpoint_path = Setters()._set_checkpoint_model()
        self.input_folder = Setters()._set_input_folder()
        self.base_data_name = Setters()._set_base_data_name()
        self.optimizer = Setters()._set_optimizer()

        self.start_epoch = int(self.training_parameters['start_epoch'])
        self.decode_type = language_aux
        self.fine_tune_encoder = fine_tune_encoder
        self.word_map = word_map
        self.device = device

        self.checkpoint_exists = False

    # load checkpoints if any
    def _load_weights_from_checkpoint(self, decoder, decoder_optimizer, encoder, encoder_optimizer, is_current_best = True):

        # Initialize / load checkpoint_model
        logging.info("saving checkpoint to {} ...".format(self.checkpoint_path if is_current_best else '../' + Setters()._set_checkpoint_model(is_best=False)))
        if os.path.exists('../' + self.checkpoint_path if is_current_best else + Setters()._set_checkpoint_model(is_best=False)):
            logging.info("checkpoint exists in %s, loading...", ' ../ ' + self.checkpoint_path if is_current_best else Setters()._set_checkpoint_model(is_best=False))
            if torch.cuda.is_available():
                checkpoint = torch.load('../' + self.checkpoint_path if is_current_best else Setters()._set_checkpoint_model(is_best=False))
            else:
                checkpoint = torch.load('../' + self.checkpoint_path if is_current_best else Setters()._set_checkpoint_model(is_best=False),
                                        map_location=torch.device("cpu"))

            # load optimizers and start epoch
            self.start_epoch = checkpoint['epoch'] + 1
            self.epochs_since_improvement = checkpoint['epochs_since_improvement']

            # load loss and bleu4
            self.best_bleu4 = checkpoint['bleu-4']
            # self.checkpoint_val_loss = checkpoint['val_loss']

            # load weights for encoder,decoder
            decoder.load_state_dict(checkpoint['decoder'])
            decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer'])
            encoder.load_state_dict(checkpoint['encoder'])
            if self.fine_tune_encoder:
                encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
            else:
                encoder_optimizer = None

            self.checkpoint_exists = True

            if self.fine_tune_encoder is True and encoder_optimizer is not None:
                print("fine tuning encoder...")
                encoder.fine_tune(self.fine_tune_encoder)
                self.encoder_optimizer = self.optimizer._get_optimizer(
                    params=filter(lambda p: p.requires_grad, encoder.parameters()),
                    lr=float(self.training_parameters['encoder_lr']))

        else:
            logging.info(
                "No checkpoint. Will start model from beggining\n")

    def _setup_dataloaders(self):

        # Custom dataloaders
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        self.train_loader = torch.utils.data.DataLoader(
            CaptionDataset(self.input_folder, self.base_data_name, self.decode_type, 'TRAIN',
                           transform=transforms.Compose([normalize])),
            batch_size=int(self.training_parameters['batch_size']), shuffle=True,
            num_workers=int(self.training_parameters['workers'])
            , pin_memory=True)
        self.val_loader = torch.utils.data.DataLoader(
            CaptionDataset(self.input_folder, self.base_data_name, self.decode_type, 'VAL',
                           transform=transforms.Compose([normalize])),
            batch_size=int(self.training_parameters['batch_size']), shuffle=True,
            num_workers=int(self.training_parameters['workers']),
            pin_memory=True)

    def _setup_train(self, train_method, validate_method):

        self.early_stopping = EarlyStopping(
            epochs_limit_without_improvement=int(self.training_parameters['epochs_limit_without_improv']),
            epochs_since_last_improvement=self.epochs_since_improvement
            if self.checkpoint_exists else 0,
            baseline=self.best_bleu4 if self.checkpoint_exists else 0,
            encoder_optimizer=self.encoder_optimizer,
            decoder_optimizer=self.decoder_optimizer,
            period_decay_lr=int(self.training_parameters['period_decay_lr'])
            , mode='metric')  # after x periods, decay the learning rate
        #
        for epoch in range(self.start_epoch, int(self.training_parameters['epochs'])):
        #
            self.current_epoch = epoch

            if self.early_stopping.is_to_stop_training_early():
                break

            train_method(train_loader=self.train_loader,
                         encoder=self.encoder,
                         decoder=self.decoder,
                         criterion=self.criterion,
                         encoder_optimizer=self.encoder_optimizer,
                         decoder_optimizer=self.decoder_optimizer,
                         epoch=epoch,
                         print_freq=int(self.training_parameters['print_freq']),
                         device=self.device)

            # One epoch's validation
            self.recent_bleu4 = validate_method(val_loader=self.val_loader,
                                                encoder=self.encoder,
                                                decoder=self.decoder,
                                                criterion=self.criterion,
                                                device=self.device)
            # Check if there was an improvement
            self.early_stopping.check_improvement(self.recent_bleu4)

            # Save checkpoint

            self._save_checkpoint(self.early_stopping.is_current_val_best(),
                                  epoch, self.early_stopping.get_number_of_epochs_without_improvement(),
                                  self.encoder, self.decoder, self.encoder_optimizer,
                                  self.decoder_optimizer, self.recent_bleu4)

    def _save_checkpoint(self, val_loss_improved, epoch, epochs_without_improvement, encoder, decoder,
                         encoder_optimizer,
                         decoder_optimizer, bleu4):

        """
        Saves model checkpoint.
        :param epoch: epoch number
        :param epochs_without_improvement: number of epochs since last improvement in BLEU-4 score
        :param encoder: encoder model
        :param decoder: decoder model
        :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
        :param decoder_optimizer: optimizer to update decoder's weights
        :param bleu4: validation BLEU-4 score for this epoch
        """

        if val_loss_improved:
            state = {'epoch': epoch,
                     'epochs_since_improvement': epochs_without_improvement,
                     'bleu-4': bleu4,
                     'encoder': encoder.state_dict(),
                     'decoder': decoder.state_dict(),
                     'encoder_optimizer': encoder_optimizer.state_dict() if self.fine_tune_encoder else None,
                     'decoder_optimizer': decoder_optimizer.state_dict()}

            filename_best_checkpoint = '../' + self.checkpoint_path
            torch.save(state, filename_best_checkpoint)
            logging.info("Saved checkpoint")

        else:
            logging.info("Not best checkpoint, saving anyways...")
            state = {'epoch': epoch,
                     'epochs_since_improvement': epochs_without_improvement,
                     'bleu-4': bleu4,
                     'encoder': encoder.state_dict(),
                     'decoder': decoder.state_dict(),
                     'encoder_optimizer': encoder_optimizer.state_dict() if self.fine_tune_encoder else None,
                     'decoder_optimizer': decoder_optimizer.state_dict()}

            filename_checkpoint = '../' + Setters()._set_checkpoint_model(is_best=False)
            torch.save(state, filename_checkpoint)
            logging.info("Saved checkpoint")



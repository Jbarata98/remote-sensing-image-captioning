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
    def _load_weights_from_checkpoint(self, decoder, decoder_optimizer, encoder, encoder_optimizer):

        # Initialize / load checkpoint_model
        logging.info("saving checkpoint to {} ...".format(self.checkpoint_path))
        if os.path.exists('../' + self.checkpoint_path):
            logging.info("checkpoint exists in %s, loading...", ' ../ ' + self.checkpoint_path)
            if torch.cuda.is_available():
                checkpoint = torch.load('../' + self.checkpoint_path)
            else:
                checkpoint = torch.load('../' + self.checkpoint_path,
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
                                                device=self.device,
                                                word_map=self.word_map if self.decode_type == AUX_LMs.GPT2.value else None,
                                                vocab_size=self.vocab_size)
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

    # @staticmethod
    # def _train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch, print_freq,
    #            device):
    #     """
    #         Performs one epoch's training.
    #         :param encoder: encoder model
    #         :param decoder: decoder model
    #         :param criterion: loss layer
    #         :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    #         :param decoder_optimizer: optimizer to update decoder's weights
    #         :param epoch: epoch number
    #         """
    #
    #     decoder.train()  # train mode (dropout and batchnorm is used)
    #     encoder.train()
    #
    #     batch_time = AverageMeter()  # forward prop. + back prop. time
    #     data_time = AverageMeter()  # data loading time
    #     losses = AverageMeter()  # loss (per word decoded)
    #     top5accs = AverageMeter()  # top5 accuracy
    #
    #     start = time.time()
    #
    #     # Batches
    #     for i, (imgs, caps, caplens) in enumerate(train_loader):
    #         data_time.update(time.time() - start)
    #
    #         # Move to GPU, if available
    #         imgs = imgs.to(device)
    #         caps = caps.to(device)
    #         caplens = caplens.to(device)
    #         # Forward prop.
    #         imgs = encoder(imgs)
    #
    #         scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)
    #         # print("got the scores")
    #
    #         # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
    #         targets = caps_sorted[:, 1:]
    #
    #         # Remove timesteps that we didn't decode at, or are pads
    #         # pack_padded_sequence is an easy trick to do this
    #         scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
    #         targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
    #         # print("un-padded them")
    #         # Calculate loss
    #         loss = criterion(scores, targets)
    #         # print("calculated the loss")
    #         # Add doubly stochastic attention regularization
    #         loss += float(h_parameter['alpha_c']) * ((1. - alphas.sum(dim=1)) ** 2).mean()
    #         # print("added loss")
    #         # Back prop.
    #         decoder_optimizer.zero_grad()
    #
    #         if encoder_optimizer is not None:
    #             encoder_optimizer.zero_grad()
    #         loss.backward()
    #         # print("back-propagated")
    #         # Clip gradients
    #         if float(h_parameter['grad_clip']) is not None:
    #             clip_gradient(decoder_optimizer, float(h_parameter['grad_clip']))
    #             if encoder_optimizer is not None:
    #                 clip_gradient(encoder_optimizer, float(h_parameter['grad_clip']))
    #         # print("clipped-gradients")
    #
    #         # Update weights
    #         decoder_optimizer.step()
    #         if encoder_optimizer is not None:
    #             print("updating weights for encoder...")
    #             encoder_optimizer.step()
    #         print("updated weights", i)
    #         # Keep track of metrics
    #         top5 = accuracy(scores, targets, 5)
    #         losses.update(loss.item(), sum(decode_lengths))
    #         top5accs.update(top5, sum(decode_lengths))
    #         batch_time.update(time.time() - start)
    #
    #         start = time.time()
    #
    #         # Print status
    #         if i % print_freq == 0:
    #             print('Epoch: [{0}][{1}/{2}]\t'
    #                   'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
    #                   'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
    #                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
    #                   'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
    #                                                                           batch_time=batch_time,
    #                                                                           data_time=data_time, loss=losses,
    #                                                                           top5=top5accs))
    # #
    # def _validate(self, val_loader, encoder, decoder, criterion, device, word_map=None, vocab_size=None):
    #     """
    #      Performs one epoch's validation.
    #      :param val_loader: DataLoader for validation data.
    #      :param encoder: encoder model
    #      :param decoder: decoder model
    #      :param criterion: loss layer
    #      :return: BLEU-4 score
    #      """
    #     decoder.eval()  # eval mode (no dropout or batchnorm)
    #     if encoder is not None:
    #         encoder.eval()
    #
    #     batch_time = AverageMeter()
    #     losses = AverageMeter()
    #     top5accs = AverageMeter()
    #
    #     start = time.time()
    #
    #     references = list()  # references (true captions) for calculating BLEU-4 score
    #     hypotheses = list()  # hypotheses (predictions)
    #
    #     # explicitly disable gradient calculation to avoid CUDA memory error
    #     # solves the issue #57
    #     with torch.no_grad():
    #         # Batches
    #         for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):
    #
    #             # Move to device, if available
    #             imgs = imgs.to(device)
    #             caps = caps.to(device)
    #             caplens = caplens.to(device)
    #
    #             # Forward prop.
    #             if encoder is not None:
    #                 imgs = encoder(imgs)
    #                 scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)
    #                 # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
    #                 targets = caps_sorted[:, 1:]
    #
    #                 # Remove timesteps that we didn't decode at, or are pads
    #                 # pack_padded_sequence is an easy trick to do this
    #                 scores_copy = scores.clone()
    #                 scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
    #                 targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
    #
    #                 # Calculate loss
    #                 loss = criterion(scores, targets)
    #
    #                 # Add doubly stochastic attention regularization
    #                 loss += float(h_parameter['alpha_c']) * ((1. - alphas.sum(dim=1)) ** 2).mean()
    #
    #                 # Keep track of metrics
    #                 losses.update(loss.item(), sum(decode_lengths))
    #                 top5 = accuracy(scores, targets, 5)
    #                 top5accs.update(top5, sum(decode_lengths))
    #                 batch_time.update(time.time() - start)
    #
    #                 start = time.time()
    #
    #             if i % int(h_parameter['print_freq']) == 0:
    #                 print('Validation: [{0}/{1}]\t'
    #                       'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
    #                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
    #                       'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader),
    #                                                                                 batch_time=batch_time,
    #                                                                                 loss=losses, top5=top5accs))
    #
    #         # Store references (true captions), and hypothesis (prediction) for each image
    #         # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    #         # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
    #
    #         # References
    #
    #         allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
    #         for j in range(allcaps.shape[0]):
    #             img_caps = allcaps[j].tolist()
    #             print("img_caps:", img_caps)
    #             # decode
    #             if self.decode_type == AUX_LMs.GPT2.value and not CUSTOM_VOCAB:  # needs to use as wordpiece - auxLM tokenizer
    #                 img_captions = list(
    #                     map(lambda c: [w for w in c if
    #                                    w not in {AuxLM_tokenizer.bos_token_id, AuxLM_tokenizer.pad_token_id}],
    #                         img_caps))  # remove <start> and pads
    #
    #             # full vocab
    #             if self.decode_type == AUX_LMs.GPT2.value and CUSTOM_VOCAB:
    #                 img_captions = list(
    #                     map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
    #                         img_caps))  # remove <start> and pads
    #
    #             # baseline uses unks
    #             else:
    #                 img_captions = list(map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<unk>'],
    #                                                                              word_map['<pad>']}],
    #                                         img_caps))  # remove <start> and pads
    #             references.append(img_captions)
    #
    #             # Hypotheses
    #         _, preds = torch.max(scores_copy, dim=2)
    #         preds = preds.tolist()
    #         temp_preds = list()
    #         for j, p in enumerate(preds):
    #             temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
    #         preds = temp_preds
    #         hypotheses.extend(preds)
    #
    #         assert len(references) == len(hypotheses)
    #
    #         # Calculate BLEU-4 scores
    #         bleu4 = corpus_bleu(references, hypotheses)
    #
    #         print(
    #             '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
    #                 loss=losses,
    #                 top5=top5accs,
    #                 bleu=bleu4))
    #
    #         return bleu4

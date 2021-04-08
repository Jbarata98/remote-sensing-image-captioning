from encoderdecoder_scripts.abstract_encoder import Encoder
from encoderdecoder_scripts.baseline.base_AttentionModel import LSTMWithAttention
from configs.initializers import *

from configs.get_training_details import *

# training details on file configs/training_details.txt

global best_bleu4, epochs_since_improvement, start_epoch


class TrainEndToEnd:
    """
    Training and validation of the model.
    """

    def __init__(self, decoder_type, fine_tune_encoder=False, checkpoint=checkpoint_model, word_map=None,
                 data=data_name, device=DEVICE):

        self.decode_type = decoder_type
        self.checkpoint_model = checkpoint
        self.fine_tune_encoder = fine_tune_encoder
        self.word_map = word_map
        self.data_name = data
        self.device = device

    # setup vocabulary
    def _setup_vocab(self):
        if self.decode_type == AUX_LMs.GPT2.value:
            self.vocab_size = AuxLM_tokenizer.vocab_size

        else:
            # Read word map(for baseline)
            word_map_file = os.path.join(data_folder, 'WORDMAP_' + self.data_name + '.json')
            with open(word_map_file, 'r') as j:
                self.word_map = json.load(j)
                self.vocab_size = len(self.word_map)

    # setup models (encoder,decoder and AuxLM for fusion)
    def _init_models(self):

        self.decoder = LSTMWithAttention(attention_dim=int(h_parameter['attention_dim']),
                                         embed_dim=int(h_parameter['emb_dim']),
                                         decoder_dim=int(h_parameter['decoder_dim']),
                                         vocab_size=self.vocab_size,
                                         dropout=float(h_parameter['dropout']))

        self.decoder_optimizer = OPTIMIZERS._get_optimizer(OPTIMIZER)(
            params=filter(lambda p: p.requires_grad, self.decoder.parameters()),
            lr=float(h_parameter['decoder_lr']))

        self.aux_LM = AuxLM_model  # defined in utils

        self.encoder = Encoder(model_type=ENCODER_MODEL, fine_tune=self.fine_tune_encoder)
        self.encoder.fine_tune(self.fine_tune_encoder)

        self.encoder_optimizer = OPTIMIZERS._get_optimizer(OPTIMIZER)(
            params=filter(lambda p: p.requires_grad, self.encoder.parameters()),
            lr=float(h_parameter['encoder_lr'])) if self.fine_tune_encoder else None

        # Move to GPU, if available

        self.decoder = self.decoder.to(self.device)
        self.encoder = self.encoder.to(self.device)

        # Loss function
        self.criterion = OPTIMIZERS._get_loss_function(LOSS)

    # load checkpoints if any
    def _load_weights_from_checkpoint(self):

        # Initialize / load checkpoint_model
        if os.path.exists(PATHS._get_checkpoint_path()):
            logging.info("checkpoint exists, loading...")
            if torch.cuda.is_available():
                checkpoint = torch.load(PATHS._get_checkpoint_path())
            else:
                checkpoint = torch.load(PATHS._get_checkpoint_path(), map_location=torch.device("cpu"))

            # load optimizers and start epoch
            self.start_epoch = checkpoint['epoch'] + 1
            self.epochs_since_improvement = checkpoint['epochs_since_improvement']

            # load loss and bleu4
            self.best_bleu4 = checkpoint['bleu-4']
            self.checkpoint_val_loss = checkpoint['val_loss']

            # load weights for encoder,decoder
            self.decoder = checkpoint['decoder']
            self.decoder_optimizer = checkpoint['decoder_optimizer']
            self.encoder = checkpoint['encoder']
            self.encoder_optimizer = checkpoint['encoder_optimizer']

            if self.fine_tune_encoder is True and self.encoder_optimizer is None:
                print("fine tuning encoder...")
                self.encoder.fine_tune(self.fine_tune_encoder)
                self.encoder_optimizer = OPTIMIZERS._get_optimizer(OPTIMIZER)(
                    params=filter(lambda p: p.requires_grad, self.encoder.parameters()),
                    lr=float(h_parameter['encoder_lr']))

        else:
            logging.info(
                "No checkpoint. Will start model from beggining\n")

    # Custom dataloaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TRAIN', transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'VAL', transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    # Epochs
    for epoch in range(start_epoch, epochs):

        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 6:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 2 == 0:
            print("DECODER:")
            adjust_learning_rate(decoder_optimizer, 0.8)
            if fine_tune_encoder:
                print("ENCODER:")
                adjust_learning_rate(encoder_optimizer, 0.8)

        # One epoch's training
        train(train_loader=train_loader,
              encoder=encoder,
              decoder=decoder,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch)

        # One epoch's validation
        recent_bleu4 = validate(val_loader=val_loader,
                                encoder=encoder,
                                decoder=decoder,
                                criterion=criterion)

        # Check if there was an improvement
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, recent_bleu4, is_best)

# def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
#     """
#     Performs one epoch's training.
#     :param train_loader: DataLoader for training data
#     :param encoder: encoder model
#     :param decoder: decoder model
#     :param criterion: loss layer
#     :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
#     :param decoder_optimizer: optimizer to update decoder's weights
#     :param epoch: epoch number
#     """
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
#         print(imgs.shape)
#         # Forward prop.
#         imgs = encoder(imgs)
#         print(imgs.shape)
#         scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)
#         # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
#         targets = caps_sorted[:, 1:]
#
#         # Remove timesteps that we didn't decode at, or are pads
#         # pack_padded_sequence is an easy trick to do this
#         scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
#         targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
#         print(scores.shape)
#         print(targets.shape)
#         # Calculate loss
#         loss = criterion(scores, targets)
#
#         # Add doubly stochastic attention regularization
#         loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
#
#         # Back prop.
#         decoder_optimizer.zero_grad()
#         if encoder_optimizer is not None:
#             encoder_optimizer.zero_grad()
#         loss.backward()
#
#         # Clip gradients
#         if grad_clip is not None:
#             clip_gradient(decoder_optimizer, grad_clip)
#             if encoder_optimizer is not None:
#                 clip_gradient(encoder_optimizer, grad_clip)
#
#         # Update weights
#         decoder_optimizer.step()
#         if encoder_optimizer is not None:
#             encoder_optimizer.step()
#
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
#
#
# def validate(val_loader, encoder, decoder, criterion):
#     """
#     Performs one epoch's validation.
#     :param val_loader: DataLoader for validation data.
#     :param encoder: encoder model
#     :param decoder: decoder model
#     :param criterion: loss layer
#     :return: BLEU-4 score
#     """
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
#             scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)
#
#             # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
#             targets = caps_sorted[:, 1:]
#
#             # Remove timesteps that we didn't decode at, or are pads
#             # pack_padded_sequence is an easy trick to do this
#             scores_copy = scores.clone()
#             scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
#             targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
#
#             # Calculate loss
#             loss = criterion(scores, targets)
#
#             # Add doubly stochastic attention regularization
#             loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
#
#             # Keep track of metrics
#             losses.update(loss.item(), sum(decode_lengths))
#             top5 = accuracy(scores, targets, 5)
#             top5accs.update(top5, sum(decode_lengths))
#             batch_time.update(time.time() - start)
#
#             start = time.time()
#
#             if i % print_freq == 0:
#                 print('Validation: [{0}/{1}]\t'
#                       'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                       'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader),
#                                                                                 batch_time=batch_time,
#                                                                                 loss=losses, top5=top5accs))
#
#             # Store references (true captions), and hypothesis (prediction) for each image
#             # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
#             # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
#
#             # References
#             allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
#             for j in range(allcaps.shape[0]):
#                 img_caps = allcaps[j].tolist()
#                 img_captions = list(
#                     map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
#                         img_caps))  # remove <start> and pads
#                 references.append(img_captions)
#
#             # Hypotheses
#             _, preds = torch.max(scores_copy, dim=2)
#             preds = preds.tolist()
#             temp_preds = list()
#             for j, p in enumerate(preds):
#                 temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
#             preds = temp_preds
#             hypotheses.extend(preds)
#
#             assert len(references) == len(hypotheses)
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
#     return bleu4

import json
import time

from nltk.translate.bleu_score import corpus_bleu
from torch.nn.utils.rnn import pack_padded_sequence

from src.abstract_train import AbstractTrain
from src.configs.setters.set_initializers import *
from src.captioning_scripts.abstract_encoder import Encoder
from src.captioning_scripts.fusion.pegasus.decoder_pegasus_simple import PegasusFusionWithAttention


class TrainPegasus(AbstractTrain):
    """
    training and validation of Pegasus fusion model
    """

    def __init__(self, language_aux, fine_tune_encoder=False, device=DEVICE, nr_inputs=1):

        super().__init__(language_aux, fine_tune_encoder, device)

        self.start_epoch = int(self.training_parameters['start_epoch'])
        self.fine_tune_encoder = fine_tune_encoder
        self.device = device
        self.decode_type = language_aux
        self.checkpoint_exists = False
        self.aux_lm = Setters()._set_aux_lm(pretrain = True)
        self.nr_inputs = nr_inputs

        self.sim_mapping_file = Setters()._set_paths()._get_similarity_mapping_path(nr_similarities=self.nr_inputs)

        # sentence max len = ( max_len * nr_captions per image) * nr_inputs( how many similar images retrieving) + 1 (end_token)
        self.sentence_max_len = (int(self.training_parameters['max_cap_length']) * int(
            self.training_parameters['captions_per_image'])) * self.nr_inputs + 1

        pegasus_input = os.path.join(self.input_folder, 'TRAIN_PEGASUS_INPUT_.json')
        with open(pegasus_input, 'r') as j:
            # print(pegasus_input)
            self.pegasus_input = json.load(j)

    def _setup_vocab(self):

        if not CUSTOM_VOCAB:
            logging.info("setting up vocab for " + self.decode_type)
            self.vocab_size = len(self.aux_lm["tokenizer"])

        else:

            logging.info("setting up custom vocab ...")

            # main name of word-map file'
            word_map_file = os.path.join(self.input_folder, 'WORDMAP_' + self.base_data_name + '.json')

            # load the word-map file
            with open(word_map_file, 'r') as j:
                self.word_map = json.load(j)
                self.vocab_size = len(self.word_map)

            hashmap_file = os.path.join(self.input_folder, 'PEGASUS_HASHMAP_' + self.base_data_name + '.json')

            with open(hashmap_file, 'r') as j:
                self.hashmap = json.load(j)

        img_similarity = os.path.join('../' + self.sim_mapping_file)

        with open(img_similarity, 'r') as j:
            self.sim_mapping = json.load(j)

        return self.hashmap, self.sim_mapping

    def _init_model(self):
        print("initializing decoder with {} auxiliary language model...".format(self.decode_type))

        self.decoder = PegasusFusionWithAttention(aux_lm=self.aux_lm
                                                  , aux_dim=int(self.training_parameters['auxLM_dim'])
                                                  , attention_dim=int(
                self.training_parameters['attention_dim']),
                                                  embed_dim=int(self.training_parameters['emb_dim']),
                                                  decoder_dim=int(self.training_parameters['decoder_dim']),
                                                  vocab=self.word_map,
                                                  hashmap=self.hashmap,
                                                  vocab_size=self.vocab_size,
                                                  sim_mapping=self.sim_mapping,
                                                  max_len=self.sentence_max_len,
                                                  dropout=float(self.training_parameters['dropout']))

        self.decoder.fine_tune_pegasus(fine_tune=False)

        self.decoder_optimizer = self.optimizer._get_optimizer(
            params=filter(lambda p: p.requires_grad, self.decoder.parameters()),
            lr=float(self.training_parameters['decoder_lr']))

        self.encoder = Encoder(model_type=ENCODER_MODEL, fine_tune=self.fine_tune_encoder)
        self.encoder.fine_tune(self.fine_tune_encoder)

        self.encoder_optimizer = self.optimizer._get_optimizer(OPTIMIZER)(
            params=filter(lambda p: p.requires_grad, self.encoder.parameters()),
            lr=float(self.training_parameters['encoder_lr'])) if self.fine_tune_encoder else None

        # Move to GPU, if available
        self.decoder = self.decoder.to(self.device)
        self.encoder = self.encoder.to(self.device)

        # Loss function
        self.criterion = self.optimizer._get_loss_function()

        return self.decoder, self.encoder

    def _train(self, train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch,
               print_freq,
               device):
        """
            Performs one epoch's training.
            :param encoder: encoder model
            :param decoder: decoder model
            :param criterion: loss layer
            :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
            :param decoder_optimizer: optimizer to update decoder's weights
            :param epoch: epoch number
            """

        decoder.train()  # train mode (dropout and batchnorm is used)
        encoder.train()

        batch_time = AverageMeter()  # forward prop. + back prop. time
        data_time = AverageMeter()  # data loading time
        losses = AverageMeter()  # loss (per word decoded)
        top5accs = AverageMeter()  # top5 accuracy

        start = time.time()

        # Batches
        for i, (imgs, paths, caps, caplens) in enumerate(train_loader):
            data_time.update(time.time() - start)

            # Move to GPU, if available
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)
            # Forward prop.
            imgs = encoder(imgs)

            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, paths, caps, caplens,
                                                                            self.pegasus_input)
            # print("got the scores")

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
            # print("un-padded them")
            # Calculate loss
            if scores.size(0) != targets.size(0):
                print("error in these paths:", paths)

            loss = criterion(scores, targets)
            # print("calculated the loss")
            # Add doubly stochastic attention regularization
            loss += float(self.training_parameters['alpha_c']) * ((1. - alphas.sum(dim=1)) ** 2).mean()
            # print("added loss")
            # Back prop.
            decoder_optimizer.zero_grad()

            if encoder_optimizer is not None:
                encoder_optimizer.zero_grad()
            loss.backward()
            # print("back-propagated")
            # Clip gradients
            if float(self.training_parameters['grad_clip']) is not None:
                clip_gradient(decoder_optimizer, float(self.training_parameters['grad_clip']))
                if encoder_optimizer is not None:
                    clip_gradient(encoder_optimizer, float(self.training_parameters['grad_clip']))
            # print("clipped-gradients")

            # Update weights
            decoder_optimizer.step()
            if encoder_optimizer is not None:
                print("updating weights for encoder...")
                encoder_optimizer.step()
            print("updated weights", i)
            # Keep track of metrics
            top5 = accuracy(scores, targets, 5)
            losses.update(loss.item(), sum(decode_lengths))
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            # Print status
            if i % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                              batch_time=batch_time,
                                                                              data_time=data_time, loss=losses,
                                                                              top5=top5accs))

    def _validate(self, val_loader, encoder, decoder, criterion, device):
        """
         Performs one epoch's validation.
         :param val_loader: DataLoader for validation data.
         :param encoder: encoder model
         :param decoder: decoder model
         :param criterion: loss layer
         :return: BLEU-4 score
         """

        decoder.eval()  # eval mode (no dropout or batchnorm)
        if encoder is not None:
            encoder.eval()

        batch_time = AverageMeter()
        losses = AverageMeter()
        top5accs = AverageMeter()

        start = time.time()

        references = list()  # references (true captions) for calculating BLEU-4 score
        hypotheses = list()  # hypotheses (predictions)

        # explicitly disable gradient calculation to avoid CUDA memory error
        # solves the issue #57

        with torch.no_grad():
            # Batches
            for i, (imgs, paths, caps, caplens, allcaps) in enumerate(val_loader):

                # Move to device, if available
                imgs = imgs.to(device)
                caps = caps.to(device)
                caplens = caplens.to(device)

                # Forward prop.
                if encoder is not None:
                    imgs = encoder(imgs)

                scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, paths, caps, caplens,
                                                                                self.pegasus_input)
                # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
                targets = caps_sorted[:, 1:]

                # Remove timesteps that we didn't decode at, or are pads
                # pack_padded_sequence is an easy trick to do this
                scores_copy = scores.clone()
                scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
                targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

                # Calculate loss
                loss = criterion(scores, targets)

                # Add doubly stochastic attention regularization
                loss += float(self.training_parameters['alpha_c']) * (
                        (1. - alphas.sum(dim=1)) ** 2).mean()

                # Keep track of metrics
                losses.update(loss.item(), sum(decode_lengths))
                top5 = accuracy(scores, targets, 5)
                top5accs.update(top5, sum(decode_lengths))
                batch_time.update(time.time() - start)

                start = time.time()

                if i % int(self.training_parameters['print_freq']) == 0:
                    print('Validation: [{0}/{1}]\t'
                          'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader),
                                                                                    batch_time=batch_time,
                                                                                    loss=losses,
                                                                                    top5=top5accs))

                # Store references (true captions), and hypothesis (prediction) for each image
                # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
                # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

                # References

                allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
                for j in range(allcaps.shape[0]):
                    img_caps = allcaps[j].tolist()
                    # print("img_caps:", img_caps)
                    # decode
                    if not CUSTOM_VOCAB:  # needs to use as wordpiece - auxLM tokenizer
                        img_captions = list(
                            map(lambda c: [w for w in c if
                                           w not in {self.aux_lm["tokenizer"].bos_token_id,
                                                     self.aux_lm["tokenizer"].pad_token_id}],
                                img_caps))  # remove <start> and pads

                    # full vocab
                    else:
                        img_captions = list(
                            map(lambda c: [w for w in c if w not in {self.word_map['<start>'], self.word_map['<pad>']}],
                                img_caps))  # remove <start> and pads

                    references.append(img_captions)

                    # Hypotheses
                _, preds = torch.max(scores_copy, dim=2)
                preds = preds.tolist()
                temp_preds = list()
                for j, p in enumerate(preds):
                    temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
                preds = temp_preds
                hypotheses.extend(preds)

                assert len(references) == len(hypotheses)

            # Calculate BLEU-4 scores
            bleu4 = corpus_bleu(references, hypotheses)

            print(
                '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
                    loss=losses,
                    top5=top5accs,
                    bleu=bleu4))

        return bleu4

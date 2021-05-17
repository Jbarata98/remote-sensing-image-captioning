import json
import os

from torchvision.transforms import transforms
from tqdm import tqdm

import torch.nn.functional as F
from src.abstract_eval import AbstractEvaluator
from src.configs.setters.set_initializers import *
from src.configs.utils.datasets import CaptionDataset


class EvalPegasus(AbstractEvaluator):

    def __init__(self, encoder, decoder, device, hashmap, sim_mapping, pegasus_input, checkpoint, b_size):

        super().__init__(encoder, decoder, device, checkpoint, b_size)

        self.word_map_file = os.path.join(self.input_folder, 'WORDMAP_' + self.base_data_name + '.json')
        self.aux_lm = Setters()._set_aux_lm()
        self.aux_lm_type = AUX_LM
        self.hashmap = hashmap
        self.sim_mapping = sim_mapping
        self.pegasus_input = pegasus_input

    def _setup_vocab(self):

        if not CUSTOM_VOCAB:
            self.vocab_size = len(self.aux_lm["tokenizer"])
        else:
            with open(self.word_map_file, 'r') as j:
                self.word_map = json.load(j)
                self.rev_word_map = {v: k for k, v in self.word_map.items()}
                self.vocab_size = len(self.word_map)

    def _get_special_tokens(self):
        self.special_tokens = []

        # if AUX_LM no need for unks
        for word, tok_id in self.word_map.items():
            if word in ['<start>', '</s>', '<pad>']:
                self.special_tokens.append(tok_id)

    def _setup_evaluate(self):

        """
       Evaluation
       :return: reference and candidate scores
        """

        # Normalization transform
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

        # setup loader
        self.loader = torch.utils.data.DataLoader(
            CaptionDataset(self.input_folder, self.base_data_name, self.aux_lm_type, 'TEST',
                           transform=transforms.Compose([self.normalize])),
            batch_size=1, shuffle=False, num_workers=1)

        # Therefore, do not use a batch_size greater than 1 - IMPORTANT!

        # Lists to store references (true captions), and hypothesis (prediction) for each image
        # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
        # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

        self.references = list()
        self.hypotheses = list()


    def _evaluate(self):
        # iterate through images, paths and captions (associated with the images/paths)
        for i, (image, path, caps, caplens, allcaps) in enumerate(
                tqdm(self.loader, desc="EVALUATING AT BEAM SIZE " + str(self.beam_size))):

            k = self.beam_size

            # Move to GPU device, if available
            image = image.to(self.device)  # (1, 3, 256, 256)

            # Encode
            encoder_out = self.encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
            enc_image_size = encoder_out.size(1)
            encoder_dim = encoder_out.size(3)

            # Flatten encoding
            encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
            num_pixels = encoder_out.size(1)

            # We'll treat the problem as having a batch size of k
            encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

            # Tensor to store top k previous words at each step; now they're just <start>

            encoder_input_ids = torch.LongTensor(
                [self.pegasus_input.get(self.sim_mapping.get(path)['Most similar'])] * k).to(self.device)

            decoder_input_ids = torch.LongTensor(
                [[self.aux_lm["model"].config.decoder_start_token_id]] * k).to(self.device)

            if not CUSTOM_VOCAB:
                k_prev_words = torch.LongTensor([[self.aux_lm["model"].config.decoder_start_token_id]] * k).to(
                    self.device)
            else:
                k_prev_words = torch.LongTensor([[self.word_map['<start>']]] * k).to(self.device)
            # Tensor to store top k sequences; now they're just <start>
            seqs = k_prev_words  # (k, 1)

            # Tensor to store top k sequences' scores; now they're just 0
            top_k_scores = torch.zeros(k, 1).to(self.device)  # (k, 1)

            # Lists to store completed sequences and scores
            complete_seqs = list()
            complete_seqs_scores = list()

            pegasus_init_outputs = self.decoder.init_pegasus_encoder(encoder_input_ids, decoder_input_ids)

            # Start decoding
            step = 0
            h, c = self.decoder.init_hidden_state(encoder_out)

            # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
            while True:

                embeddings = self.decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

                awe, _ = self.decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

                gate = self.decoder.sigmoid(self.decoder.f_beta(h))  # gating scalar, (s, encoder_dim)

                awe = gate * awe

                h_auxLM = self.decoder.calc_auxLM(pegasus_init_outputs, decoder_input_ids, len(decoder_input_ids), step)
                h, c = self.decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

                h_fusion = torch.cat([h, h_auxLM], axis=-1)

                scores = self.decoder.fc(h_fusion)  # (s, vocab_size)
                scores = F.log_softmax(scores, dim=1)

                # Add
                scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

                # For the first step, all k points will have the same scores (since same k previous words, h, c)
                if step == 0:
                    top_k_scores, top_k_words = scores[0].topk(k, 0)  # (s)
                else:
                    # Unroll and find top scores, and their unrolled indices
                    top_k_scores, top_k_words = scores.view(-1).topk(k, 0)  # torch.topk(scores,k)

                # print(top_k_words)

                # Convert unrolled indices to actual indices of scores
                prev_word_ids = top_k_words // self.vocab_size  # (s)
                next_word_ids = top_k_words % self.vocab_size  # (s)
                # Add new words to sequences
                seqs = torch.cat([seqs[prev_word_ids], next_word_ids.unsqueeze(1)], dim=1)  # (s, step+1)

                # Which sequences are incomplete (didn't reach <end>)?

                if not CUSTOM_VOCAB:
                    incomplete_ids = [ind for ind, next_word in enumerate(next_word_ids) if
                                      next_word != self.aux_lm["model"].config.eos_token_id]
                else:
                    incomplete_ids = [ind for ind, next_word in enumerate(next_word_ids) if
                                      next_word != self.word_map['</s>']]

                complete_ids = list(set(range(len(next_word_ids))) - set(incomplete_ids))

                # Set aside complete sequences
                if len(complete_ids) > 0:
                    complete_seqs.extend(seqs[complete_ids].tolist())
                    complete_seqs_scores.extend(top_k_scores[complete_ids])
                k -= len(complete_ids)  # reduce beam length accordingly

                # Proceed with incomplete sequences
                if k == 0:
                    break

                seqs = seqs[incomplete_ids]
                h = h[prev_word_ids[incomplete_ids]]
                c = c[prev_word_ids[incomplete_ids]]
                encoder_out = encoder_out[prev_word_ids[incomplete_ids]]
                top_k_scores = top_k_scores[incomplete_ids].unsqueeze(1)
                k_prev_words = next_word_ids[incomplete_ids].unsqueeze(1)

                # decoder_input_ids = [[[self.hashmap.get(str(x.item()))]] for x in seqs]

                # Break if things have been going on too long
                if step > 40:
                    break
                step += 1

            i = complete_seqs_scores.index(max(complete_seqs_scores))
            seq = complete_seqs[i]

            # References
            img_caps = allcaps[0].tolist()

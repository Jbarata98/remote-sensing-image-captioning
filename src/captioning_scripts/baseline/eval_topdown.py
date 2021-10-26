from torchvision.transforms import transforms
from tqdm import tqdm

import torch.nn.functional as F
from src.abstract_eval import AbstractEvaluator
from src.configs.setters.set_initializers import *
from src.configs.utils.datasets import CaptionDataset


class EvalBaselineTopDown(AbstractEvaluator):
    """
    eval baseline with topdown attention settings
    """

    def __init__(self, encoder, decoder, aux_lm, device, word_map, vocab_size, checkpoint, b_size):

        super().__init__(encoder, decoder, device, checkpoint, b_size)

        self.word_map_file = os.path.join(self.input_folder, 'WORDMAP_' + self.base_data_name + '.json')
        # baseline uses no auxiliary language model
        self.aux_lm = aux_lm
        self.word_map = word_map
        self.rev_word_map = {v: k for k, v in self.word_map.items()}
        self.vocab_size = vocab_size
        self.encoder = encoder
        self.decoder = decoder

    def _get_special_tokens(self):
        self.special_tokens = []

        # if AUX_LM no need for unks
        for word, tok_id in self.word_map.items():
            if word in ['<start>', '<end>', '<pad>', '<unk>']:
                self.special_tokens.append(tok_id)
        return self.special_tokens

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
            CaptionDataset(self.input_folder, self.base_data_name, self.aux_lm, 'TEST',
                           transform=transforms.Compose([self.normalize])),
            batch_size=1, shuffle=False, num_workers=1)

        # Therefore, do not use a batch_size greater than 1 - IMPORTANT!

        # Lists to store references (true captions), and hypothesis (prediction) for each image
        # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
        # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

        self.hypotheses = list()

    def _evaluate(self):

        self.decoder.to(self.device)
        self.encoder.to(self.device)

        self.decoder.eval()
        self.encoder.eval()
        # For each image
        for i, (image, caps, caplens, allcaps) in enumerate(
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

            # mean-pooled image feature for input to the topdown lstm

            # Tensor to store top k previous words at each step; now they're just <start>

            k_prev_words = torch.LongTensor([[self.word_map['<start>']]] * k).to(self.device)  # (k, 1)

            # Tensor to store top k sequences; now they're just <start>
            seqs = k_prev_words  # (k, 1)

            # Tensor to store top k sequences' scores; now they're just 0
            top_k_scores = torch.zeros(k, 1).to(self.device)  # (k, 1)

            # Lists to store completed sequences and scores
            complete_seqs = list()
            complete_seqs_scores = list()

            # Start decoding
            step = 0
            h_language, c_language = self.decoder.init_hidden_state(encoder_out)
            h_topdown, c_topdown = self.decoder.init_hidden_state(encoder_out)

            # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
            while True:
                encoder_out_mean_pool = encoder_out.mean(dim=1)
                embeddings = self.decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)
                # print(encoder_out_mean_pool.shape)
                # print(embeddings.shape)
                # print(h_language.shape)
                h_topdown, c_topdown = self.decoder.TopDownLSTM(
                    torch.cat([embeddings, encoder_out_mean_pool, h_language], dim=1), (h_topdown, c_topdown))

                awe, _ = self.decoder.attention(encoder_out, h_topdown)  # (s, encoder_dim), (s, num_pixels)

                gate = self.decoder.sigmoid(self.decoder.f_beta(h_topdown))  # gating scalar, (s, encoder_dim)

                awe = gate * awe

                h_language, c_language = self.decoder.LanguageLSTM(torch.cat([h_topdown, awe], dim=1),
                                                                   (h_language, c_language))  # (s, decoder_dim)

                scores = self.decoder.fc(h_language)  # (s, vocab_size)
                scores = F.log_softmax(scores, dim=1)

                # Add
                scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

                # print(scores.shape)
                # For the first step, all k points will have the same scores (since same k previous words, h, c)
                if step == 0:
                    top_k_scores, top_k_words = scores[0].topk(k, 0)  # (s)
                else:
                    # Unroll and find top scores, and their unrolled indices
                    top_k_scores, top_k_words = scores.view(-1).topk(k, 0)  # torch.topk(scores,k)

                # print(top_k_words)

                # Convert unrolled indices to actual indices of scores
                prev_word_inds = top_k_words // self.vocab_size  # (s)
                next_word_inds = top_k_words % self.vocab_size  # (s)
                # Add new words to sequences
                # print(seqs[prev_word_inds])
                seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
                # print(next_word_inds)
                # for seq in seqs:
                #     print(AuxLM_tokenizer.decode(seq, skip_special_tokens = True))
                # Which sequences are incomplete (didn't reach <end>)?
                if TOKENIZER == TOKENIZATION.SIMPLE.value:
                    incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                                       next_word != self.word_map['<end>']]
                else:
                    if not CUSTOM_VOCAB:
                        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                                           next_word != self.aux_lm["model"].config.eos_token_id]
                    else:
                        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                                           next_word != self.word_map['</s>']]
                complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

                # Set aside complete sequences
                if len(complete_inds) > 0:
                    complete_seqs.extend(seqs[complete_inds].tolist())
                    complete_seqs_scores.extend(top_k_scores[complete_inds])
                k -= len(complete_inds)  # reduce beam length accordingly

                # Proceed with incomplete sequences
                if k == 0:
                    break

                seqs = seqs[incomplete_inds]
                h_language = h_language[prev_word_inds[incomplete_inds]]
                c_language = c_language[prev_word_inds[incomplete_inds]]

                h_topdown = h_topdown[prev_word_inds[incomplete_inds]]
                c_topdown = c_topdown[prev_word_inds[incomplete_inds]]

                encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
                top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
                k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)
                # Break if things have been going on too long
                if step > 50:
                    break
                step += 1
            # print(complete_seqs)

            i = complete_seqs_scores.index(max(complete_seqs_scores))
            seq = complete_seqs[i]

            # using full vocab

            # Hypotheses
            if TOKENIZER == TOKENIZATION.SIMPLE.value:
                self.hypotheses.append(
                    ' '.join(self.rev_word_map[w] for w in seq if w not in self._get_special_tokens()))
            elif TOKENIZER == TOKENIZATION.PEGASUS.value:
                if not CUSTOM_VOCAB:

                    # Hypotheses
                    self.hypotheses.append(self.aux_lm["tokenizer"].decode(seq, skip_special_tokens=True))

                    # using AUXLM and CUSTOM_VOCAB
                else:
                    # print("seq is:\n", seq)
                    # print("token" , [self.rev_word_map[w] for w in seq if
                    #                                 w not in self._get_special_tokens()])
                    #
                    self.hypotheses.append(
                        self.aux_lm["tokenizer"].convert_tokens_to_string([self.rev_word_map[w] for w in seq if
                                                                           w not in self._get_special_tokens()]))

            # print(hypotheses)

        with open('../' + Setters()._set_paths()._get_hypothesis_path(results_array=True), "wb") as f:
            pickle.dump(self.hypotheses, f)

        return self.hypotheses
#

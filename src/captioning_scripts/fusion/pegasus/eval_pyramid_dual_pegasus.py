import src
import torch.nn.functional as F
from bert_score import BERTScorer
from tqdm import tqdm
from torchvision.transforms import transforms
from src.abstract_eval import AbstractEvaluator
from src.configs.setters.set_initializers import *
from src.configs.utils.datasets import CaptionDataset
from src.metrics_files.bleurt import score as bleurt_sc
from src.configs.getters.get_data_paths import bleurt_checkpoint

class EvalPyramidPegasus(AbstractEvaluator):
    """
    class to Eval (Pyramid Feature Maps + Dual Attention) Pegasus Fusion Architecture
    """

    def __init__(self, encoder, decoder, aux_lm, device, hashmap, word_map, vocab_size, sim_mapping, pegasus_input,
                 checkpoint, b_size):

        super().__init__(encoder, decoder, device, checkpoint, b_size)

        self.word_map_file = os.path.join(self.input_folder, 'WORDMAP_' + self.base_data_name + '.json')
        self.aux_lm = aux_lm
        self.aux_lm_type = AUX_LM
        self.hashmap = hashmap
        self.sim_mapping = sim_mapping
        self.pegasus_input = pegasus_input
        self.word_map = word_map
        self.rev_word_map = {v: k for k, v in self.word_map.items()}
        self.vocab_size = vocab_size
        self.decoder = decoder
        self.encoder = encoder
        self.cider_scorer = src.metrics_files.pycocoevalcap.cider.cider.Cider()
        self.bleu_scorer = src.metrics_files.pycocoevalcap.bleu.bleu.Bleu()
        self.meteor_scorer = src.metrics_files.pycocoevalcap.meteor.meteor.Meteor()
        self.bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)
        self.bleurt_scorer = bleurt_sc.BleurtScorer(bleurt_checkpoint)

        
    def _get_special_tokens(self):

        self.special_tokens = []

        # if AUX_LM no need for unks
        for word, tok_id in self.word_map.items():
            if word in ['<start>', '</s>', '<pad>']:
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
            CaptionDataset(self.input_folder, self.base_data_name, self.aux_lm_type, 'TEST',
                           transform=transforms.Compose([self.normalize])),
            batch_size=1, shuffle=False, num_workers=1)

        # Therefore, do not use a batch_size greater than 1 - IMPORTANT!

        # Lists to store references (true captions), and hypothesis (prediction) for each image
        # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
        # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

        self.hypotheses = list()

    def get_minimum_bayes_risk( self, complete_seqs, beam_scores, num_candidates=2 ):
        if num_candidates is None: num_candidates=len(beam_scores)
        aux = []
        scores = []
        for seq_aux in complete_seqs:
            if not CUSTOM_VOCAB: aux.append(self.aux_lm["tokenizer"].decode(seq_aux, skip_special_tokens=True))
            else: aux.append(self.aux_lm["tokenizer"].convert_tokens_to_string([self.rev_word_map[w] for w in seq_aux if w not in self._get_special_tokens()]))
        best_k = np.argpartition(beam_scores, -num_candidates)[-num_candidates:]
        for pos in best_k:
            seq = aux[pos]
            score = 0
            num_score = 0
            for pos2, seq2 in enumerate(aux):
                if pos == pos2: continue
                targets = { 1: [ seq2 ] }
                samples = { 1: [ seq ] }
                bleurts = self.bleurt_scorer.score( references=[seq], candidates=[seq2], batch_size=None)
                #_, _, bertss = self.bert_scorer.score([seq],[seq2])
                #_, ciders = self.cider_scorer.compute_score(targets, samples)
                #_, bleus = self.bleu_scorer.compute_score(targets, samples)
                #_, meteors = self.meteor_scorer.compute_score(targets, samples)
                score += bleurts[0] * beam_scores[pos2]
                num_score += beam_scores[pos2]
            scores.append(score / num_score)
        return best_k[np.argmax(scores)]        
        
    def _evaluate(self):
        self.decoder.to(self.device)
        self.encoder.to(self.device)
        self.beam_size = 25 # hack... remove this and set the self.beam_size parameter for the experiment accordingly
        self.decoder.eval()
        self.encoder.eval()
        # iterate through images, paths and captions (associated with the images/paths)
        for i, (image, path, caps, caplens, allcaps) in enumerate(
                tqdm(self.loader, desc="EVALUATING AT BEAM SIZE " + str(self.beam_size))):

            k = self.beam_size

            # Move to GPU device, if available
            image = image.to(self.device)  # (1, 3, 256, 256)

            # Encode
            encoder_outputs = self.encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)

            # print("LEN", encoder_outputs[0].shape, encoder_outputs[1].shape, encoder_outputs[2].shape)
            # print(encoder_out.shape)
            if PYRAMID_REDUCTION_LAYER:
                encoder_out = torch.cat((encoder_outputs[0], encoder_outputs[1], encoder_outputs[2]), 1)
                # encoder output linearly projected
                pyramid_concat = encoder_out.permute(0, 2, 1)
                # print(pyramid_concat.shape)
                # reduce concatenated maps from (I_1 + I_2 + I_3) dims to (I_1)

                pyramid_concat = self.decoder.pyramid_reduction(self.decoder.relu(pyramid_concat))
                # print(pyramid_concat.shape)
                pyramid_concat = pyramid_concat.permute(0, 2, 1)

                encoder_out = pyramid_concat
                # print(pyramid_concat.shape)
            else:
                encoder_out = encoder_outputs

            # print(encoder_out.shape)
            # We'll treat the problem as having a batch size of k
            encoder_dim = encoder_out.size(2)

            num_pixels = encoder_out.size(1)

            encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)
            # print(encoder_out.shape)

            # Tensor to ids for encoder (similar captions)
            # print(path)
            encoder_input_ids = torch.LongTensor([self.pegasus_input.get(
                self.sim_mapping.get(path[0])['Most similar(s)' if MULTI_INPUT else 'Most similar'])] * k).to(
                self.device)

            # Tensor to store top k previous words at each step; now they're just <start>
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

                v_s, _ = self.decoder.spatial_attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)
                v_c = self.decoder.channel_attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

                awe = v_s + v_c

                h_auxLM = self.decoder.calc_auxLM(pegasus_init_outputs, decoder_input_ids, len(decoder_input_ids), step)
                h, c = self.decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

                if REDUCTION_LAYER:
                    h_auxLM = self.decoder.reduction_layer(self.decoder.relu(h_auxLM))

                if CONCAT_ONLY:
                    h_fusion = torch.cat([h, h_auxLM], axis=-1)
                if FUSION == 'simple':
                    h_cat = torch.cat([h, h_auxLM], axis=-1)
                    h_fusion = self.decoder.projection_layer(self.decoder.relu(h_cat))
                elif FUSION == 'cold':
                    # considering the h_auxLM was already reduced (reduction_layer)
                    h_cat = torch.cat([h, h_auxLM], axis=-1)
                    # print(h_lstm.shape, h_auxLM.shape)
                    h_projected = self.decoder.init_projection_layer(self.decoder.relu(h_cat))
                    # print(h_projected.shape)
                    h_cold_fusion = torch.cat([h, (torch.mul(h_projected, h_auxLM))], axis=-1)
                    h_fusion = self.decoder.final_projection_layer(self.decoder.relu(h_cold_fusion))
                    # print(h_cfusion.shape)
                    # h_fusion =
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

                # convert ids for aux_LM calculation
                decoder_input_ids = torch.stack(
                    [torch.LongTensor([[self.hashmap.get(str(tok_id)) for tok_id in seq]]) for seq in
                     seqs.tolist()]).to(self.device)

                # Break if things have been going on too long
                if step > 40:
                    break
                step += 1

            i = complete_seqs_scores.index(max(complete_seqs_scores))
            # Alternative: Use minimum risk Bayes decoding
            i = self.get_minimum_bayes_risk( complete_seqs, complete_seqs_scores )            
            seq = complete_seqs[i]

            # References
            img_caps = allcaps[0].tolist()
            # print("word_map", self.rev_word_map)

            # using full vocab
            if not CUSTOM_VOCAB:

                # Hypotheses
                self.hypotheses.append(self.aux_lm["tokenizer"].decode(seq, skip_special_tokens=True))

            # using AUXLM and CUSTOM_VOCAB
            else:
                # print("seq is:\n", seq)
                # print("token" , [self.rev_word_map[w] for w in seq if
                #                                 w not in self._get_special_tokens()])
                #
                self.hypotheses.append(self.aux_lm["tokenizer"].convert_tokens_to_string([self.rev_word_map[w] for w in seq if
                                                w not in self._get_special_tokens()]))

        with open('../' + Setters()._set_paths()._get_hypothesis_path(results_array=True), "wb") as f:
            pickle.dump(self.hypotheses, f)

        return self.hypotheses
    #

import torch.optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import transforms

from src.configs.datasets import CaptionDataset
from src.configs.initializers import *
from tqdm import tqdm

cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead


class evaluator:

    def __init__(self, device, word_map=None, b_size=3):
        # Parameters
        # sets device for model and PyTorch tensors

        self.device = device
        self.beam_size = b_size
        self.word_map = word_map

    # Load model
    def _load_checkpoint(self):
        print(f"loading checkpoint in {checkpoint_model}")
        if torch.cuda.is_available():

            self.checkpoint = torch.load('../' + checkpoint_model)
        # cpu if not using colab
        else:
            self.checkpoint = torch.load('../' + checkpoint_model, map_location=torch.device('cpu'))

        self.decoder = self.checkpoint['decoder']
        decoder = self.decoder.to(self.device)
        decoder.eval()

        self.encoder = self.checkpoint['encoder']
        encoder = self.encoder.to(self.device)
        # encoder is frozen
        # encoder.eval()

        # Load word map (word2id)
        if AuxLM == AUX_LMs.GPT2.value and not CUSTOM_VOCAB:
            self.vocab_size = len(AuxLM_tokenizer)
        # baseline
        else:
            with open(self.word_map, 'r') as j:
                self.word_map = json.load(j)
                self.rev_word_map = {v: k for k, v in self.word_map.items()}
                self.vocab_size = len(self.word_map)

    def _get_special_tokens(self, w_map, aux_LM = False):
        special_tokens = []
        for word,id in w_map.items():
            #if AUX_LM no need for unks
            if AuxLM:
                if word in ['<start>', '<end>', '<pad>']:
                    special_tokens.append(id)
            #using UNKS with no AuxLM
            else:
                if word in ['<start>', '<end>','<unk>', '<pad>']:
                    special_tokens.append(id)

        # print("special tokens found:", special_tokens)
        # print("discarding...")
        return special_tokens


    def _evaluate(self):

        """
        Evaluation
        :return: reference and candidate scores
        """

        # Normalization transform
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        # DataLoader
        loader = torch.utils.data.DataLoader(
            CaptionDataset(data_folder, data_name, 'TEST', transform=transforms.Compose([normalize])),
            batch_size=1, shuffle=False, num_workers=1)

        # TODO: Batched Beam Search
        # Therefore, do not use a batch_size greater than 1 - IMPORTANT!

        # Lists to store references (true captions), and hypothesis (prediction) for each image
        # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
        # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

        references = list()
        hypotheses = list()


        # For each image
        for i, (image, caps, caplens, allcaps) in enumerate(
                tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(self.beam_size))):

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
            if AuxLM == AUX_LMs.GPT2.value and not CUSTOM_VOCAB:
                k_prev_words = torch.LongTensor([[AuxLM_tokenizer.bos_token_id]] * k).to(self.device)  # (k, 1)
            else:
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
            h, c = self.decoder.init_hidden_state(encoder_out)

            # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
            while True:

                embeddings = self.decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

                awe, _ = self.decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

                gate = self.decoder.sigmoid(self.decoder.f_beta(h))  # gating scalar, (s, encoder_dim)

                awe = gate * awe

                h_auxLM = self.decoder.calc_auxLM(seqs, len(seqs),step, eval = True)
                h, c = self.decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

                h_fusion = torch.cat([h,h_auxLM], axis=-1)

                scores = self.decoder.fc(h_fusion)  # (s, vocab_size)
                scores = F.log_softmax(scores, dim=1)

                # Add
                scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

                #print(scores.shape)
                # For the first step, all k points will have the same scores (since same k previous words, h, c)
                if step == 0:
                    top_k_scores, top_k_words = scores[0].topk(k, 0)  # (s)
                else:
                    # Unroll and find top scores, and their unrolled indices
                    top_k_scores, top_k_words =  scores.view(-1).topk(k,0) #torch.topk(scores,k)

                #print(top_k_words)

                # Convert unrolled indices to actual indices of scores
                prev_word_inds = top_k_words // self.vocab_size  # (s)
                next_word_inds = top_k_words % self.vocab_size # (s)
                # Add new words to sequences

                seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

                # for seq in seqs:
                #     print(AuxLM_tokenizer.decode(seq, skip_special_tokens = True))
                # Which sequences are incomplete (didn't reach <end>)?
                if AuxLM == AUX_LMs.GPT2 and not CUSTOM_VOCAB :
                    incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                                       next_word != AuxLM_tokenizer.eos_token_id]
                else:
                    incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                                       next_word != self.word_map['<end>']]
                    
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
                h = h[prev_word_inds[incomplete_inds]]
                c = c[prev_word_inds[incomplete_inds]]
                encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
                top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
                k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

                # Break if things have been going on too long
                if step > 40:
                    break
                step += 1

            i = complete_seqs_scores.index(max(complete_seqs_scores))
            seq = complete_seqs[i]

            # References
            img_caps = allcaps[0].tolist()
            #using full vocab
            if AuxLM == AUX_LMs.GPT2.value and not CUSTOM_VOCAB:

                img_captions = list(list(AuxLM_tokenizer.decode(cap,  skip_special_tokens = True) for cap in img_caps))


                references.append(img_captions)
                # Hypotheses
                hypotheses.append(AuxLM_tokenizer.decode(seq,  skip_special_tokens = True))

            #baseline
            elif AuxLM == None and not CUSTOM_VOCAB:

                img_captions = list(
                    map(lambda c: [
                        ' '.join(self.rev_word_map[w] for w in c if w not in self._get_special_tokens(self.word_map))],
                        img_caps))  # remove <start> and pads
                references.append(img_captions)
                # Hypotheses
                hypotheses.append(' '.join( self.rev_word_map[w] for w in seq if w not in self._get_special_tokens(self.word_map)))
                # print(hypotheses)

            #using AUXLM and CUSTOM_VOCAB
            else:
                img_captions = list(
                    map(lambda c: [' '.join(self.rev_word_map[w] for w in c if w not in self._get_special_tokens(self.word_map))],
                        img_caps))  # remove <start> and pads
                references.append(img_captions)
                # Hypotheses
                hypotheses.append(' '.join(AuxLM_tokenizer.decode(AuxLM_tokenizer.convert_tokens_to_ids(self.rev_word_map[w])) for w in seq if w not in self._get_special_tokens(self.word_map)))
            # print(hypotheses)
            assert len(references) == len(hypotheses)

        with open('../' + PATHS._get_results_path(results_array=True), "wb") as f:
            pickle.dump(references, f)

        with open('../' + PATHS._get_hypothesis_path(results_array=True), "wb") as f:
            pickle.dump(hypotheses, f)

        return references, hypotheses
    #
import math

from src.captioning_scripts.baseline.train_baseline import TrainBaseline
from src.captioning_scripts.fusion.pegasus.train_pegasus import TrainPegasus
from src.configs.setters.set_initializers import  *
from src.captioning_scripts.baseline.dual_AttentionModel import LSTMWithPyramidAttention
from src.captioning_scripts.abstract_encoder import Encoder

import torch
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
import argparse
from PIL import Image
import cv2

RESHAPE_CONCAT = True

if ATTENTION == ATTENTION_TYPE.pyramid_attention.value:
    pyramid = True
else:
    pyramid = False

def caption_image_beam_search(encoder, decoder, image_path, word_map, beam_size=3):
    """
    Reads an image and captions it with beam search.
    :param encoder: encoder model
    :param decoder: decoder model
    :param image_path: path to image
    :param word_map: word map
    :param beam_size: number of sequences to consider at each decode-step
    :return: caption, weights for visualization
    """

    k = beam_size
    vocab_size = len(word_map)

    # Read image and process
    print(image_path)
    img = cv2.imread(image_path)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)

    img = cv2.resize(img, (256, 256))

    img = img.transpose(2, 0, 1)

    img = img / 255.

    img = torch.FloatTensor(img).to(DEVICE)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([normalize])
    image = transform(img)  # (3, 256, 256)



    # Encode
    image = image.unsqueeze(0)  # (1, 3, 256, 256)
    encoder_outputs = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)

    # print(encoder_out[0].shape,encoder_out[1].shape,encoder_out[2].shape)

    alphas_pyramid = []
    seqs_pyramid = []
    for encoder_out in encoder_outputs:
        # print(encoder_out.shape)
        k = beam_size
        if ATTENTION == ATTENTION_TYPE.soft_attention.value:
            encoder_dim = encoder_out.size(2)
            img_size = encoder_out.size(1)

        elif ATTENTION == ATTENTION_TYPE.pyramid_attention.value:
            encoder_dim = encoder_out.size(3)
            img_size = encoder_out.size(1)


        # Flatten encoding
        # encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)

        # Flatten encoding
        # encoder_out = encoder_out.flatten(start_dim=1, end_dim=2)

        encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # We'll treat the problem as having a batch size of k
        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)
        print(encoder_out.shape)

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(DEVICE)  # (k, 1)
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(DEVICE)  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>

        # Tensor to store top k sequences' alphas; now they're just 1s
        seqs_alpha = torch.ones(k, 1, img_size, img_size).to(DEVICE)  # (k, 1, enc_image_size, enc_image_size)

        # Lists to store completed sequences, their alphas and scores
        complete_seqs = list()
        complete_seqs_alpha = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
        h, c = decoder.init_hidden_state(encoder_out)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:

            embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)
            if pyramid:
                v_s, alpha = decoder.spatial_attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)
                v_c,beta = decoder.channel_attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)
                # print(alpha.shape)
                # print(beta.mean(dim=2).shape)
                # alpha = alpha*beta.mean(dim=2)
                # print("alpha", alpha)
                awe = v_s + v_c
            # soft
            else:
                awe, alpha = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

                gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)

                awe = gate * awe


            # print(alpha.shape)

            alpha = alpha.view(-1, img_size, img_size)  # (s, enc_image_size, enc_image_size)

            # gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
            # awe = gate * awe

            h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

            scores = decoder.fc(h)  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=1)

            # Add
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words // vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)

            # Add new words to sequences, alphas
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
            seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],
                                   dim=1)  # (s, step+1, enc_image_size, enc_image_size)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != word_map['</s>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            seqs_alpha = seqs_alpha[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > 50:
                break
            step += 1

        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]
        alphas = complete_seqs_alpha[i]
        # print(alphas)
        seqs_pyramid.append(seq),alphas_pyramid.append(alphas)
        print(seqs_pyramid)
    return seqs_pyramid, alphas_pyramid


def visualize_att(image_path, seq, alphas, rev_word_map,save_name, smooth=True):
    """
    Visualizes caption with weights at every word.
    Adapted from paper authors' repo: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb
    :param image_path: path to image that has been captioned
    :param seq: caption
    :param alphas: weights
    :param rev_word_map: reverse word mapping, i.e. ix2word
    :param smooth: smooth weights?
    """
    image = Image.open(image_path)
    image = image.resize([12*24, 12*24])

    words = Setters()._set_aux_lm(pretrain=False)["tokenizer"].convert_tokens_to_string([rev_word_map[w] for w in seq])
    words = words.split(' ')

    print(words)
    print("len_words", len(words))

    if RESHAPE_CONCAT:
        alpha_new = []
        for alpha in alphas:
            alpha_new.append(torch.FloatTensor(alpha))

        new_words_alphas = []
        for feat_map_alpha in alpha_new:
            feat_maps_reshape = []
            for words_alpha in feat_map_alpha:
                # print(words_alpha.shape)
                # out = F.interpolate(words_alpha, size=8)
                # print(words_alpha.shape
                # x = torch.randn(1, 3, 4, 4)
                # print(words_alpha.shape)
                words_alpha = F.interpolate(words_alpha.unsqueeze(0).unsqueeze(0), size = (8,8)).squeeze(0).squeeze(0)
                feat_maps_reshape.append(words_alpha)
            # new reshaped feature maps
            new_words_alphas.append(feat_maps_reshape)

        avg_alphas = []
        feat_maps_avg = []
        # print(len(new_words_alphas[0]),len(new_words_alphas[1]),len(new_words_alphas[2]))
        # in each feat map
        for (feat_map1,feat_map2,feat_map3) in zip(new_words_alphas[0],new_words_alphas[1],new_words_alphas[2]):
            new_avg = torch.rand(8, 8)
            for i,(row_x,row_y,row_z)in enumerate(zip(feat_map1,feat_map2,feat_map3)):
                # print(i)
        #         # print(feat_map1.shape)
                for z,(value_x,value_y,value_z) in enumerate(zip(row_x, row_y,row_z)):
                    # print(z)
        #             # print(value_y,value_x,value_z)
                    new_avg[i][z] = (value_x + value_y + value_z)/3
        #             # print(new_avg[i][z])
        #         # print(new_avg)
        #     print(new_avg)
            avg_alphas.append(new_avg)
        avg_alphas.append(feat_maps_avg)


        print(len(avg_alphas))
        print(avg_alphas[0].shape)
        print(avg_alphas[0][0].shape)
        print(avg_alphas[0][2])



        # avg_alphas = torch.FloatTensor(avg_alphas)


    for t in range(len(words)):
        if t > 50:
            break

        plt.subplot(np.ceil(len(words) /5.), 5, t + 1)
        plt.text(0, 300, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=8)
        plt.imshow(image)
        if RESHAPE_CONCAT:
            print(len(avg_alphas))
            print(t)
            current_alpha = avg_alphas[t]
        else:
            current_alpha = alphas[t, :]

        # print(len(current_alpha))
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha.numpy(), upscale=32, sigma=8)
            # print("shape", alpha.shape)
        else:
            alpha = skimage.transform.resize(current_alpha.numpy(), [14 * 24, 14 * 24])
            # print("shape", alpha.shape)

        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.7)
        plt.set_cmap(cm.Oranges)
        plt.axis('off')
    # plt.savefig('../' + Setters()._set_paths()._get_hypothesis_path(results_array=False)+'.jpg')
    plt.show()


if __name__ == '__main__':
    print("parsing")
    parser = argparse.ArgumentParser(description='Remote Sensing Image Captioning - Generate Caption')

    parser.add_argument('--imgs', nargs="+", help='path to image')
    parser.add_argument('--model', '-m', help='path to model')
    parser.add_argument('--word_map', '-wm', help='path to word map JSON')
    parser.add_argument('--beam_size', '-b', default=5, type=int, help='beam size for beam search')
    parser.add_argument('--save_name', '-s')

    parser.add_argument('--dont_smooth', dest='smooth', action='store_false', help='do not smooth alpha overlay')

    args = parser.parse_args()

    print(args)

    # Load model
    print("loading model...")
    _train = TrainBaseline(language_aux=None, fine_tune_encoder=False, model_version='v2')
    _train._setup_vocab()
    # initiate the models
    _train._init_model()
    encoder = _train.encoder
    decoder = _train.decoder
    checkpoint = torch.load(args.model, map_location=str(DEVICE))
    decoder.load_state_dict(checkpoint['decoder'])
    encoder.load_state_dict(checkpoint['encoder'])
    encoder = encoder.to(DEVICE)
    decoder = decoder.to(DEVICE)
    decoder.eval()
    encoder.eval()

    # Load word map (word2ix)
    with open(args.word_map, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}  # ix2word

    # Encode, decode with attention and beam search
    print("running...")
    for img in args.imgs:
        seqs_pyramid, alphas_pyramid = caption_image_beam_search(encoder, decoder, "../data/images/RSICD_images/" + img, word_map, args.beam_size)
    # print(len(alphas_pyramid), len(seqs_pyramid))
    # for (seqs,alpha) in zip(seqs_pyramid,alphas_pyramid):
        # alphas = torch.FloatTensor(alpha)
        if RESHAPE_CONCAT:
            shapes = [len(seqs_pyramid[0]),len(seqs_pyramid[1]),len(seqs_pyramid[2])]
            visualize_att("../data/images/RSICD_images/" + img, seqs_pyramid[np.argmin(shapes)], alphas_pyramid, rev_word_map, args.save_name, args.smooth)
        else:
            for i, (alpha, seqs) in enumerate(zip(alphas_pyramid, seqs_pyramid)):
                alpha = torch.FloatTensor(alpha)
                visualize_att("../data/images/RSICD_images/" + img, seqs, alpha, rev_word_map, args.save_name, args.smooth)
    #
        # alpha_new = []
        # for alpha in alphas_pyramid:
        #     alpha_new.append(torch.FloatTensor(alpha))
        # new_words_alphas = []
        # for feat_map_alpha in alpha_new:
        #     feat_maps_reshape = []
        #     for words_alpha in feat_map_alpha:
        #         # print(words_alpha.shape)
        #         # out = F.interpolate(words_alpha, size=8)
        #         # print(words_alpha.shape
        #         # x = torch.randn(1, 3, 4, 4)
        #         # print(words_alpha.shape)
        #         words_alpha = F.interpolate(words_alpha.unsqueeze(0).unsqueeze(0), size=(8, 8)).squeeze(0).squeeze(0)
        #         feat_maps_reshape.append(words_alpha)
        #     # new reshaped feature maps
        #     new_words_alphas.append(feat_maps_reshape)
        #
        # # print(len(new_words_alphas))

        # rgb_arrays = []
        # for i,(alpha,seqs) in enumerate(zip(alphas_pyramid,seqs_pyramid)):
        #     # print(len(alpha))
        #     # print(len(alpha[0]))
        #
        #     # print(alpha)
        #     alpha = torch.FloatTensor(alpha[0])
        #     # print(alpha[3])
        #     print(alpha.shape)
        #
        #     # data = np.zeros((h, w, 3), dtype=np.uint8)
        #     alpha*=255
        #
        #     img = Image.fromarray(alpha.numpy(), mode = "RGB")
        #
        #     plt.imshow(img)
        #     plt.show()
        #
        #      # red patch in upper left
        #     r, g, b = img.split()


        #     if i == 0:
        #         # Increase Reds
        #         r = r.point(lambda i: i * 1)
        #
        #         # Decrease Greens
        #         g = g.point(lambda i: i * 0)
        #
        #         # Decrease Blues
        #         b = b.point(lambda i: i * 0)
        #
        #     elif i ==1:
        #         # Increase Reds
        #         r = r.point(lambda i: i * 0)
        #
        #         # Decrease Greens
        #         g = g.point(lambda i: i * 1)
        #
        #         # Decrease Blues
        #         b = b.point(lambda i: i * 0)
        #
        #     elif i ==2:
        #         # Increase Reds
        #         r = r.point(lambda i: i * 0)
        #
        #         # Decrease Greens
        #         g = g.point(lambda i: i * 0)
        #
        #         # Decrease Blues
        #         b = b.point(lambda i: i * 1)
        #
        #
        #     # Recombine back to RGB image
        #     result = Image.merge('RGB', (r, g, b))
        #
        #     print(np.array(result))
        #
        #     # result.save('result.png')
        #     # print("array",np.array(result))
        # #
        # #     rgb_arrays.append(np.array(result))
        # #     # img.save('my.png')
        # #     # plt.imshow(result)
        # #     # plt.show()
        # # # print("summed",sum(rgb_arrays))
        # # matrixes_merge =[]
        # # for (r,g,b) in zip(rgb_arrays[0],rgb_arrays[1],rgb_arrays[2]):
        # #     matrixes_merge.append(np.column_stack((r[:,0],g[:,1],b[:,2])))
        # #
        # # # print(matrixes_merge)
        # #
        # # # img = Image.fromarray(matrixes_merge[0], mode="RGB")
        # # #
        # # plt.imshow(matrixes_merge[0])
        # # plt.show()
        #
        #     # visualize_att(args.img, seqs, alpha, rev_word_map, args.save_name, args.smooth)


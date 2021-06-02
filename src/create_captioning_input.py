import cv2
import json
import h5py
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample
from src.configs.utils.vocab_aux_functions import *


class InputGen:

    """
       Creates input files for training, validation, and test data.
       :param dataset: name of dataset
       :param json_path: path of JSON file with splits and captions
       :param image_folder: folder with downloaded images
       :param captions_per_image: number of captions to sample per image
       :param min_word_freq: words occurring less frequently than this threshold are binned as <unk>s
       :param output_folder: folder to save files
       :param max_len: don't sample captions longer than this length
    """

    def __init__(self, dataset, json_path, image_folder, captions_per_image, min_word_freq, output_folder,
                 max_len=30):

        self.dataset = dataset
        self.path = json_path
        self.image_folder = image_folder
        self.captions_per_image = captions_per_image
        self.min_word_freq = min_word_freq
        self.output_folder = output_folder
        self.max_len = max_len

    def _setup_input_files(self, lang_model):

        self.LM = lang_model

        # remote-sensing image captioning datasets
        assert self.dataset in {'rsicd', 'ucm', 'sydney'}

        # Create a base/root name for all output files
        base_filename = self.dataset + '_' + str(self.captions_per_image) + '_cap_per_img_' + str(
            self.min_word_freq) + '_min_word_freq'

        # Read JSON
        with open(self.path, 'r') as j:
            data = json.load(j)

        # Read image paths and captions for each image
        train_image_paths = []
        train_image_captions = []
        val_image_paths = []
        val_image_captions = []
        test_image_paths = []
        test_image_captions = []
        word_freq = Counter()

        # define tokenizer if AUX_LM != None
        if ARCHITECTURE == ARCHITECTURES.FUSION.value:
            self.tokenizer = Setters()._set_aux_lm()["tokenizer"]
        else:
            self.tokenizer = None

        for img in data['images']:
            captions = []
            for c in img['sentences']:

                word_freq, captions = save_captions(c, captions, self.LM, word_freq, self.max_len)
                if len(captions) == 0:
                    continue

            path = os.path.join(
                self.image_folder, img['filename'])

            # save captions according to each split
            if img['split'] in {'train'}:
                train_image_paths.append(path)
                train_image_captions.append(captions)
            elif img['split'] in {'val'}:
                val_image_paths.append(path)
                val_image_captions.append(captions)
            elif img['split'] in {'test'}:
                test_image_paths.append(path)
                test_image_captions.append(captions)

        # Sanity check
        assert len(train_image_paths) == len(train_image_captions)
        assert len(val_image_paths) == len(val_image_captions)
        assert len(test_image_paths) == len(test_image_captions)

        words = [w for w in word_freq.keys() if
                 word_freq[w] > self.min_word_freq]  # basically words that occur more than min word freq

        self.word_map = set_wordmap(words)

        # Save word map to a JSON
        with open(os.path.join(self.output_folder, 'WORDMAP_' + base_filename + '.json'), 'w') as j:
            json.dump(self.word_map, j)
        # #         #
        # Sample captions for each image, save images to HDF5 file, and captions and their lengths to JSON files
        seed(123)
        for impaths, imcaps, split in [(train_image_paths, train_image_captions, 'TRAIN'),
                                       (val_image_paths, val_image_captions, 'VAL'),
                                       (test_image_paths, test_image_captions, 'TEST')]:

            print(os.path.join(self.output_folder, split + '_IMAGES_' + base_filename + '.hdf5'))
            if os.path.exists(os.path.join(self.output_folder, split + '_IMAGES_' + base_filename + '.hdf5')):
                print("Already existed, rewriting...")

                os.remove(os.path.join(self.output_folder, split + '_IMAGES_' + base_filename + '.hdf5'))

            with h5py.File(os.path.join(self.output_folder, split + '_IMAGES_' + base_filename + '.hdf5'), 'a') as h:
                # Make a note of the number of captions we are sampling per image
                h.attrs['captions_per_image'] = self.captions_per_image

                # Create dataset inside HDF5 file to store images
                images = h.create_dataset('images', (len(impaths), 3, 224, 224), dtype='uint8')

                print("\nReading %s images and captions, storing to file...\n" % split)

                enc_captions = []
                caplens = []
                # only used for pegasus or retrieval-like tasks
                img_names = []
                #
                for i, path in enumerate(tqdm(impaths)):
                    # save img names for pegasus will be necessary for similarity checking
                    if AUX_LM == AUX_LMs.PEGASUS.value:
                        img_names = save_paths(path, img_names)
                    # Sample captions
                    if len(imcaps[i]) < self.captions_per_image:
                        captions = imcaps[i] + [choice(imcaps[i]) for _ in
                                                range(self.captions_per_image - len(imcaps[i]))]
                    else:
                        captions = sample(imcaps[i], k=self.captions_per_image)

                    # Sanity check
                    assert len(captions) == self.captions_per_image

                    # Read images
                    img = cv2.imread(impaths[i])

                    # if grey/white image
                    if len(img.shape) == 2:
                        print("shape is 2, image grey/white")
                        img = img[:, :, np.newaxis]
                        img = np.concatenate([img, img, img], axis=2)

                    img = cv2.resize(img, (224, 224))
                    img = img.transpose(2, 0, 1)

                    assert img.shape == (3, 224, 224)
                    assert np.max(img) <= 255

                    # Save image to HDF5 file
                    images[i] = img
                    # encode the captions
                    for j, c in enumerate(captions):
                        enc_captions, caplens = encode_captions(self.tokenizer, c, self.word_map, self.max_len, enc_captions,
                                                                caplens)
                # Sanity check
                assert images.shape[0] * self.captions_per_image == len(enc_captions) == len(caplens)

                if AUX_LM == AUX_LMs.PEGASUS.value:
                # Save paths to use (for pegasus only)
                    with open(os.path.join(self.output_folder, split + '_IMGPATHS_.json'), 'w') as j:

                        json.dump(img_names, j)
                # Save encoded captions and their lengths to JSON files
                with open(os.path.join(self.output_folder, split + '_CAPTIONS_' + base_filename + '.json'), 'w') as j:

                    json.dump(enc_captions, j)

                with open(os.path.join(self.output_folder, split + '_CAPLENS_' + base_filename + '.json'), 'w') as j:
                    json.dump(caplens, j)
        #


# Create input files (along with word map)
generate_input = InputGen(dataset=DATASET,
                          json_path=Setters()._set_paths()._get_captions_path(),  # path of the .json file with the captions
                          image_folder=Setters()._set_paths()._get_images_path(),  # folder containing the images
                          captions_per_image=5,
                          min_word_freq=int(Setters()._set_training_parameters()['min_word_freq']),
                          output_folder=Setters()._set_paths()._get_input_path(),
                          max_len=int(Setters()._set_training_parameters()['max_cap_length']))

generate_input._setup_input_files(lang_model = AUX_LM)

from configs.utils import *


class input_generator():

    """
       Creates input files for training, validation, and test data.
       :param dataset: name of dataset, one of 'coco', 'flickr8k', 'flickr30k'
       :param json_path: path of JSON file with splits and captions
       :param image_folder: folder with downloaded images
       :param captions_per_image: number of captions to sample per image
       :param min_word_freq: words occuring less frequently than this threshold are binned as <unk>s
       :param output_folder: folder to save files
       :param max_len: don't sample captions longer than this length
       """

    def __init__(self,dataset, json_path, image_folder, captions_per_image, min_word_freq, output_folder,
                       max_len=30):

        self.dataset = dataset
        self.path = json_path
        self.image_folder = image_folder
        self.captions_per_image = captions_per_image
        self.min_word_freq = min_word_freq
        self.output_folder = output_folder
        self.max_len = max_len


    def _setup_input_files(self, LM = AUX_LM):

        self.LM = LM

        assert self.dataset in {'rsicd', 'ucm', 'sydney'}

        # Create a base/root name for all output files
        base_filename = self.dataset + '_' + str(self.captions_per_image) + '_cap_per_img_' + str(
            min_word_freq) + '_min_word_freq'

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

        for img in data['images']:
            captions = []
            for c in img['sentences']:
                    # Update word frequency
                    word_freq.update(c['tokens'])
                    if len(c['tokens']) <= self.max_len:
                        if self.LM == AUX_LMs.GPT2.value:  # if its GPT2, need to save raw captions only
                            captions.append(c['raw'])
                        else:
                            captions.append(c['tokens']) # if its only an lstm (baseline)

                    if len(captions) == 0:
                        continue


            path = os.path.join(
                self.image_folder, img['filename'])

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
        #
        if LM is None:
            # we need a wordmap if dealing only with LSTM
            words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
            word_map = {k: v + 1 for v, k in enumerate(words)}
            word_map['<unk>'] = len(word_map) + 1
            word_map['<start>'] = len(word_map) + 1
            word_map['<end>'] = len(word_map) + 1
            word_map['<pad>'] = 0

            # Save word map to a JSON
            with open(os.path.join(self.output_folder, 'WORDMAP_' + base_filename + '.json'), 'w') as j:
                json.dump(word_map, j)
        #
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

                for i, path in enumerate(tqdm(impaths)):

                    # Sample captions
                    if len(imcaps[i]) < self.captions_per_image:
                        captions = imcaps[i] + [choice(imcaps[i]) for _ in range(self.captions_per_image - len(imcaps[i]))]
                    else:
                        captions = sample(imcaps[i], k=self.captions_per_image)


                    # Sanity check
                    assert len(captions) == self.captions_per_image
            # #
            #         # Read images
            #         img = cv2.imread(impaths[i])
            #         if len(img.shape) == 2:
            #             img = img[:, :, np.newaxis]
            #             img = np.concatenate([img, img, img], axis=2)
            #
            #         img = cv2.resize(img, (224, 224))
            #         img = img.transpose(2, 0, 1)
            #
            #         assert img.shape == (3, 224, 224)
            #         assert np.max(img) <= 255
            #
            #         # Save image to HDF5 file
            #         images[i] = img
            # #
                    for j, c in enumerate(captions):
                        print(j,c)
                        # if self.LM == AUX_LMs.GPT2.value:  # if its GPT2, need to encode differently
                    #         captions.append(AuxLM_tokenizer(SPECIAL_TOKENS[
                    #                                    'bos_token'] + c['raw'] +
                    #                                SPECIAL_TOKENS['eos_token'], truncation=True, max_length=35,
                    #                                padding="max_length", return_tensors="pt"))
                    #     else:
                    # else:
                    #     # Encode captions
                    #     enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                    #         word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))
                    #
                    #     # Find caption lengths
                    #     c_len = len(c) + 2
                    #
                    #     enc_captions.append(enc_c)
                    #     caplens.append(c_len)
        #
        # # #     # Sanity check
        # #     assert images.shape[0] * captions_per_image == len(enc_captions) == len(caplens)
        # #
        # #     # Save encoded captions and their lengths to JSON files
        # #     with open(os.path.join(output_folder, split + '_CAPTIONS_' + base_filename + '.json'), 'w') as j:
        # #
        # #         json.dump(enc_captions, j)
        # #
        # #     with open(os.path.join(output_folder, split + '_CAPLENS_' + base_filename + '.json'), 'w') as j:
        # #         json.dump(caplens, j)
        #

# Create input files (along with word map)
generate_input = input_generator(dataset = DATASET,
                       json_path= PATHS._get_captions_path(),  # path of the .json file with the captions
                       image_folder= PATHS._get_images_path(),  # folder containing the images
                       captions_per_image=5,
                       min_word_freq=2,
                       output_folder= PATHS._get_input_path(),
                       max_len=30)
generate_input._setup_input_files()



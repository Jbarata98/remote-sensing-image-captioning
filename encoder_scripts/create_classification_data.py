from configs.utils import *
import h5py
import sys



# def create_classification_files(dataset, json_path, image_folder, output_folder):
#
#     """
#     Creates input files for training, validation, and test data.
#     :param dataset: name of dataset, one of 'coco', 'flickr8k', 'flickr30k'
#     :param json_path: path of JSON file with splits and captions
#     :param image_folder: folder with downloaded images
#     :param output_folder: folder to save files
#     """
#     with open(json_path, 'r') as j:
#     data = json.load(j)
#     train_image_paths = []
#     train_image_captions = []
#     val_image_paths = []
#     val_image_captions = []
#     test_image_paths = []
#     test_image_captions = []
#     word_freq = Counter()
#
#     for img in data['images']:
#         captions = []
#         for c in img['sentences']:
#             # Update word frequency
#             word_freq.update(c['tokens'])
#             if len(c['tokens']) <= max_len:
#                 captions.append(c['tokens'])
#
#         if len(captions) == 0:
#             continue
#
#         path = os.path.join(
#             image_folder, img['filename'])
#
#         if img['split'] in {'train'}:
#             train_image_paths.append(path)
#             train_image_captions.append(captions)
#         elif img['split'] in {'val'}:
#             val_image_paths.append(path)
#             val_image_captions.append(captions)
#         elif img['split'] in {'test'}:
#             test_image_paths.append(path)
#             test_image_captions.append(captions)
#
#     # Sanity check
#     assert len(train_image_paths) == len(train_image_captions)
#     assert len(val_image_paths) == len(val_image_captions)
#     assert len(test_image_paths) == len(test_image_captions)
#
#     # Create word map
#     words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
#     word_map = {k: v + 1 for v, k in enumerate(words)}
#     word_map['<unk>'] = len(word_map) + 1
#     word_map['<start>'] = len(word_map) + 1
#     word_map['<end>'] = len(word_map) + 1
#     word_map['<pad>'] = 0
#
#     # Create a base/root name for all output files
#     base_filename = dataset + '_' + str(captions_per_image) + '_cap_per_img_' + str(min_word_freq) + '_min_word_freq'
#
#     # Save word map to a JSON
#     with open(os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json'), 'w') as j:
#         json.dump(word_map, j)
#
#     # Sample captions for each image, save images to HDF5 file, and captions and their lengths to JSON files
#     seed(123)
#     for impaths, imcaps, split in [(train_image_paths, train_image_captions, 'TRAIN'),
#                                    (val_image_paths, val_image_captions, 'VAL'),
#                                    (test_image_paths, test_image_captions, 'TEST')]:
#         with h5py.File(os.path.join(output_folder, split + '_IMAGES_' + base_filename + '.hdf5'), 'a') as h:
#             # Make a note of the number of captions we are sampling per image
#             h.attrs['captions_per_image'] = captions_per_image
#
#             # Create dataset inside HDF5 file to store images
#             images = h.create_dataset('images', (len(impaths), 3, 224, 224), dtype='uint8')
#
#             print("\nReading %s images and captions, storing to file...\n" % split)
#
#             enc_captions = []
#             caplens = []
#
#             for i, path in enumerate(tqdm(impaths)):
#
#                 # Sample captions
#                 if len(imcaps[i]) < captions_per_image:
#                     captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]
#                 else:
#                     captions = sample(imcaps[i], k=captions_per_image)
#
#                 # Sanity check
#                 assert len(captions) == captions_per_image
#
#                 # Read images
#                 img = cv2.imread(impaths[i])
#                 if len(img.shape) == 2:
#                     img = img[:, :, np.newaxis]
#                     img = np.concatenate([img, img, img], axis=2)
#
#                 img = cv2.resize(img, (224, 224))
#                 img = img.transpose(2, 0, 1)
#
#                 assert img.shape == (3, 224, 224)
#                 assert np.max(img) <= 255
#
#                 # Save image to HDF5 file
#                 images[i] = img
#
#                 for j, c in enumerate(captions):
#                     # Encode captions
#                     enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
#                         word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))
#
#                     # Find caption lengths
#                     c_len = len(c) + 2
#
#                     enc_captions.append(enc_c)
#                     caplens.append(c_len)
#
#             # Sanity check
#             assert images.shape[0] * captions_per_image == len(enc_captions) == len(caplens)
#
#             # Save encoded captions and their lengths to JSON files
#             with open(os.path.join(output_folder, split + '_CAPTIONS_' + base_filename + '.json'), 'w') as j:
#
#                 json.dump(enc_captions, j)
#
#             with open(os.path.join(output_folder, split + '_CAPLENS_' + base_filename + '.json'), 'w') as j:
#                 json.dump(caplens, j)

if __name__ == '__main__':
    # reutilize captions json dataset to use for classification
    with open('../' + get_captions_path(DATASET), 'r') as j:
        data = json.load(j)
    #rename captions to label
    for img in data['images']:
        img['label'] = img.pop('sentences')
    #save it

    classes_data = {}
    #open classes directory and save them to dictionary format {'image' : label,...}
    for filename in os.listdir(RSICD_CLASSES_PATH):
        f = open(RSICD_CLASSES_PATH + '/' + filename, "r")
        for image in f.readlines():
            classes_data[image.split("\n")[0]] = filename.split(".txt")[0]

    #switch the previous captions by the labels for the images
    for img in data['images']:
        for image_name, label in classes_data.items():
            if image_name == img['filename']:
                img['label'] = label
                break
    #write to json
    with open((get_classification_dataset_path(DATASET)), 'w+') as j:
        json.dump(data, j)
    # json.dump(caplens, j)


def open_images():
    with h5py.File('../classification/inputs/TRAIN_IMAGES_rsicd.hdf5', "r") as f:
        # List all groups
        print("Keys: %s" % f.keys())
        a_group_key = list(f.keys())[0]

        # Get the data
        data = list(f[a_group_key])

        print(len(data))
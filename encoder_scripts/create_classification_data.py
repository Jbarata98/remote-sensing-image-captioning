from configs.utils import *
import collections
import numpy as np
import h5py
import sys

def create_classes_json():
    # reutilize captions json dataset to use for classification
    with open('../' + get_captions_path(DATASET), 'r') as j:
        data = json.load(j)
    # rename captions to label
    for img in data['images']:
        img['label'] = img.pop('sentences')
    # save it

    classes_data = {}
    # open classes directory and save them to dictionary format {'image' : label,...}
    for filename in os.listdir(RSICD_CLASSES_PATH):
        f = open(RSICD_CLASSES_PATH + '/' + filename, "r")
        for image in f.readlines():
            classes_data[image.split("\n")[0]] = filename.split(".txt")[0]

    # switch the previous captions by the labels for the images
    for img in data['images']:
        for image_name, label in classes_data.items():
            if image_name == img['filename']:
                img['label'] = label
                break

    # write to json
    with open((get_classification_dataset_path(DATASET)), 'w+') as j:
        json.dump(data, j)

def create_classes_dict(labels):
    NR_CLASSES = len(set(labels))
    classes_dict = collections.defaultdict(list)
    for category, i in zip(set(labels), range(len(labels))):
        classes_dict[category] = i
    print(classes_dict)
    return NR_CLASSES,classes_dict

def create_classification_files(dataset, json_path, image_folder, output_folder):

    """
    Creates input files for training, validation, and test data.
    :param dataset: name of dataset, one of 'coco', 'flickr8k', 'flickr30k'
    :param json_path: path of JSON file with splits and captions
    :param image_folder: folder with downloaded images
    :param output_folder: folder to save files
    """

    with open(json_path, 'r') as j:
        data = json.load(j)
        train_image_paths = []
        train_image_labels = []
        val_image_paths = []
        val_image_labels = []
        test_image_paths = []

        test_image_labels = []

    for img in data['images']:

        path = os.path.join(
            image_folder, img['filename'])

        if img['split'] in {'train'}:
            train_image_paths.append(path)
            train_image_labels.append(img['label'])
        elif img['split'] in {'val'}:
            val_image_paths.append(path)
            val_image_labels.append(img['label'])
        elif img['split'] in {'test'}:
            test_image_paths.append(path)
            test_image_labels.append(img['label'])

    # Sanity check
    assert len(train_image_paths) == len(train_image_labels)
    assert len(val_image_paths) == len(val_image_labels)
    assert len(test_image_paths) == len(test_image_labels)

    #lets use train only to create the classes dict
    NR_CLASSES,classes_dict =create_classes_dict(train_image_labels)


    #Create a base/root name for all output files
    base_filename = dataset + '_' + 'CLASSIFICATION_dataset'

    # # Sample labels for each image, save images to HDF5 file, and labels to a JSON file
    seed(123)

    for (impaths), imlabels, split in [(train_image_paths, train_image_labels, 'TRAIN'),
                                   (val_image_paths, val_image_labels, 'VAL'),
                                   (test_image_paths, test_image_labels, 'TEST')]:
        with h5py.File(os.path.join(output_folder, split + '_IMAGES_' + base_filename + '.hdf5'), 'a') as h:
            enc_labels = []

            # Create dataset inside HDF5 file to store images
            images = h.create_dataset('images', (len(impaths), 3, 224, 224), dtype='uint8')

            print("\nReading %s images and labels, storing to file...\n" % split)

            for i, path in enumerate(tqdm(impaths)):
                enc_labels.append([classes_dict[imlabels[i]]])

                # Read images
                img = cv2.imread('../' + impaths[i])
                if len(img.shape) == 2:
                    img = img[:, :, np.newaxis]
                    img = np.concatenate([img, img, img], axis=2)

                img = cv2.resize(img, (224, 224))
                img = img.transpose(2, 0, 1)

                assert img.shape == (3, 224, 224)
                assert np.max(img) <= 255

                # Save image to HDF5 file
                images[i] = img


    #Save encoded labels to JSON files
            with open(os.path.join(output_folder, split + '_LABELS_' + base_filename + '.json'), 'w') as j:
                json.dump(enc_labels, j)



    return NR_CLASSES





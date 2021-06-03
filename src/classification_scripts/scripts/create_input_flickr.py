import collections
import json
import os
from random import seed
from tqdm import tqdm
import numpy as np
import h5py
import cv2

# hardcoded script just for testing with flickr dataset
def create_flickr_input():
    root_path = "/data/images/flickr8k"
    impaths = []
    for root, dirs, files in os.walk(root_path):
        for filename in files:
            impaths.append(root_path + '/' + filename)
    # print(impaths)
    # print(len(impaths))
    if os.path.exists(
            '/experiments/encoder/inputs/TRAIN_IMAGES_flickr8k.hdf5'):
        print("Already existed, rewriting...")
        os.remove(
            '/experiments/encoder/inputs/TRAIN_IMAGES_flickr8k.hdf5')
    with h5py.File((
                   '/home/starksultana/Documentos/MEIC/5o_ano/Tese/code/remote-sensing-image-captioning/experiments/encoder/inputs/TRAIN_IMAGES_flickr8k.hdf5'),
                   'a') as h:

        # Create dataset inside HDF5 file to store images
        images = h.create_dataset('images', (len(impaths), 3, 224, 224), dtype='uint8')

        print("\nReading %s images and labels, storing to file...\n")

        # encode images and labels
        for i, path in enumerate(tqdm(impaths)):

            # Read images
            img = cv2.imread(impaths[i])

            # might have black/white jpgs
            if len(img.shape) == 2:
                img = img[:, :, np.newaxis]
                img = np.concatenate([img, img, img], axis=2)

            img = cv2.resize(img, (224, 224))
            img = img.transpose(2, 0, 1)

            assert img.shape == (3, 224, 224)
            assert np.max(img) <= 255

            # Save image to HDF5 file
            images[i] = img


create_flickr_input()

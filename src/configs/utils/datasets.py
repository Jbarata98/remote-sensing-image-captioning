import random

import torch
from PIL import Image
from torch.utils.data import Dataset
from src.configs.globals import *
from torchvision.datasets import ImageFolder
import h5py
import json
import os
import numpy as np
from src.classification_scripts.augment import histogram_matching, TwoViewTransform
from torchvision.transforms import transforms
from src.configs.setters.set_initializers import Setters

h_parameters = Setters("encoder_training_details.txt")._set_training_parameters()

class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, data_name, aux_lm, split, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.aux_lm_type = aux_lm
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Open hdf5 file where images are stored
        self.h = h5py.File(os.path.join(data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')
        self.imgs = self.h['images']

        # Captions per image
        self.cpi = self.h.attrs['captions_per_image']

        # Load encoded captions (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)

        # Load caption lengths (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)

        # load paths if using PEGASUS
        if self.aux_lm_type == AUX_LMs.PEGASUS.value:
            with open(os.path.join(data_folder, self.split + '_IMGPATHS_.json'), 'r') as j:
                self.paths = json.load(j)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.captions)

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image

        if self.aux_lm_type == AUX_LMs.PEGASUS.value:
            path = self.paths[i//self.cpi]
            # print(path)

        img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)
        if self.transform is not None:
            img = self.transform(img)

        caption = torch.LongTensor(self.captions[i])

        caplen = torch.LongTensor([self.caplens[i]])
        if self.split == 'TRAIN':
            # if using pegasus return the paths
            if self.aux_lm_type == AUX_LMs.PEGASUS.value:
                return img, path, caption, caplen
            else:
                return img, caption, caplen
        else:
            # For validation or testing, also return all 'captions_per_image' captions to find BLEU-4 score
            all_captions = torch.LongTensor(
                self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            # if using pegasus return the paths
            if self.aux_lm_type == AUX_LMs.PEGASUS.value:
                return img, path, caption, caplen, all_captions
            else:
                return img, caption, caplen, all_captions
    def __len__(self):
        return self.dataset_size


class ClassificationDataset(CaptionDataset):

    """
    Dataset class for classification task on remote sensing datasets
    """

    def __init__(self, data_folder, data_name, split, continuous=False, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        :param continuous: continuous input (one-hot)
        """
        self.split = split
        self.continuous = continuous
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Open hdf5 file where images are stored
        self.h = h5py.File(os.path.join(data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')
        self.imgs = self.h['images']

        # load target images for histogram matching if dealing with training data
        if self.split == 'TRAIN':
            self.target_h = h5py.File(os.path.join(data_folder, 'TEST_IMAGES_' + data_name + '.hdf5'), 'r')
            self.target_imgs = self.target_h['images']
        # Load encoded labels (completely into memory)
        with open(os.path.join(data_folder, self.split + '_LABELS_' + data_name + '.json'), 'r') as j:
            self.labels = json.load(j)
        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of data-points
        self.dataset_size = len(self.labels)

    def __getitem__(self, i):

        img = torch.FloatTensor(self.imgs[i] / 255.)
        if self.transform is not None:
            if h_parameters["MULTI_VIEW_BATCH"]:
                multi_view_transf = TwoViewTransform(self.transform, self.split, self.target_imgs)
                imgs_view = multi_view_transf(img)
            else:
                # regular transformations
                img = self.transform(img)

                # if dealing with training data also take into consideration transposition and randomized histogram_matching
                if self.split == 'TRAIN':
                    # randomized histogram
                    if random.choice([0, 1]) == 0:
                        img = histogram_matching(img, self.target_imgs)


        # if you want to turn the vector to one hot encoding (continuous output)
        if self.continuous:
            one_hot = np.zeros(max(self.labels)[0] + 1)
            one_hot[self.labels[i][0]] = 1
            label = torch.LongTensor(one_hot)
            if h_parameters["MULTI_VIEW_BATCH"]:
                return imgs_view, label
            else:
                return img, label

        # use discrete label
        else:
            label = torch.LongTensor(self.labels[i])
            if h_parameters["MULTI_VIEW_BATCH"]:
                return imgs_view, label
            else:
                return img, label


class FeaturesDataset(CaptionDataset):
    """
    Dataset to extract features only
    """

    def __init__(self, data_folder, data_name, split, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Open hdf5 file where images are stored
        # lowercase so the splits will be equal to the ones in the caption .json dataset
        self.h = h5py.File(os.path.join(data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')
        self.imgs = self.h['images']

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        self.dataset_size = len(self.imgs)

    def __getitem__(self, i):
        img = torch.FloatTensor(self.imgs[i] / 255.)
        if self.transform is not None:
            img = self.transform(img)
            return img


class ImageFolderWithPaths(ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    Source: https://gist.github.com/andrewjong/6b02ff237533b3b2c554701fb53d5c4d
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


class TrainRetrievalDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.data_folder = data_folder

        with open(os.path.join(data_folder, "TRAIN" + '_IMGPATHS_' + DATASET + '.json'), 'r') as j:
            self.imgpaths = json.load(j)

        # self.imgpaths=self.imgpaths[:10]
        # print("self images", self.imgpaths)
        ##TODO:REMOVE

        # Total number of datapoints
        self.dataset_size = len(self.imgpaths)
        # print("this is the actual len on begin init", self.dataset_size)

        self.transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                             transforms.RandomVerticalFlip(), transforms.RandomRotation(90),
                                             transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                         std=[0.229, 0.224, 0.225])])

    def __getitem__(self, i):
        # print(self.imgpaths[i])
        img = Image.open(self.imgpaths[i])
        img = self.transform(img)
        # print("i of retrieval dataset",i)
        return img, i

    def __len__(self):
        # print("this is the actual len on __len", self.dataset_size)
        return self.dataset_size

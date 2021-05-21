import os
import random

import h5py
import numpy as np
import torch
import torchvision.transforms.functional as TF
from matplotlib import pyplot as plt
from skimage.exposure import match_histograms

from src.configs.setters.set_initializers import Setters


class CustomRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)


def histogram_matching(ref_img, target_imgs):
    """ Perform histogram matching on a given training image """

    nr_target = list(range(len(target_imgs)))
    random_target_img = random.choice(nr_target)

    # need to convert reference to numpy
    reference = ref_img.numpy()

    target = target_imgs[random_target_img]/255

    matched = match_histograms(reference, target)

    # UNCOMMENT FOR VISUALIZATION
    # reference = np.transpose(reference,(1,2,0))
    # target = np.transpose(target, (1, 2, 0))
    # matched_temp = np.transpose(matched, (1,2,0))
    # fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),
    #                                     sharex=True, sharey=True)
    # for aa in (ax1, ax2, ax3):
    #     aa.set_axis_off()
    #
    # ax1.imshow(reference)
    # ax1.set_title('reference')
    # ax2.imshow(target)
    # ax2.set_title('target')
    # ax3.imshow(matched_temp)
    # ax3.set_title('Matched')
    #
    # plt.tight_layout()
    # plt.show()

    return torch.FloatTensor(matched)


# TEST
#
# input_folder = Setters(file='../configs/setters/training_details.txt')._set_input_folder()
# base_data_name = Setters(file='../configs/setters/training_details.txt')._set_base_data_name()
# # Open hdf5 file where images are stored
# h = h5py.File(os.path.join('../' + input_folder, 'TRAIN_IMAGES_' + base_data_name + '.hdf5'), 'r')
# imgs = h['images']
#
# h = h5py.File(os.path.join('../' +input_folder, 'TEST_IMAGES_' + base_data_name + '.hdf5'), 'r')
# target_imgs = h['images']
#
# ref = torch.FloatTensor(imgs[10] /255)
#
# histogram_matching(ref,target_imgs)
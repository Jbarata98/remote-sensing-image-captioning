import json
import os

from matplotlib import pyplot


def visualize_fmap(feature_list):
    """
    function to visualize the feature maps
    """
    square = 8
    for fmap in feature_list:
        # plot all 64 maps in an 8x8 squares
        ix = 1
        for _ in range(square):
            for _ in range(square):
                # specify subplot and turn of axis
                ax = pyplot.subplot(square, square, ix)
                ax.set_xticks([])
                ax.set_yticks([])
                # plot filter channel in grayscale
                pyplot.imshow(fmap[0, :, :, ix - 1], cmap='gray')
                ix += 1
        # show the figure
        pyplot.show()


def flatten_maps(feature_list):
    f_maps = []
    # flatten the feature maps
    for fmaps in feature_list:
        for fmap in fmaps:
            fmap = fmap.flatten(start_dim=0, end_dim=1)  # (1,7,7,2048) feature map
            fmap = fmap.mean(dim=0)

            f_maps.append(fmap)  # (2048) dimension feature map

    return f_maps


def get_image_name(path, split, dataset='remote_sensing'):
    # get captions path to retrieve image name
    train_filenames = []

    if dataset == 'remote_sensing':
        file = open(path._get_classification_dataset_path())
        data = json.load(file)
        for image in data['images']:
            if image['split'] == split:
                train_filenames.append([image['filename'], image['label']])

    # using another dataset by chance (flickr,coco,etc)
    else:
        file = "/home/starksultana/Documentos/MEIC/5o_ano/Tese/code/remote-sensing-image-captioning/data/images/flickr8k"
        for root, dirs, files in os.walk(file):
            for filename in files:
                train_filenames.append(file + '/' + filename)
    return train_filenames

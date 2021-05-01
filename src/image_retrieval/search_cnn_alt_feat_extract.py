import pickle

import cv2
import faiss
import numpy as np
from PIL import Image
import torch
from matplotlib import pyplot as plt
from torchvision import transforms
from src.configs.get_models import *
from src.configs.get_data_paths import *
# from src.configs.datasets import ImageFolderWithPaths

from src.configs.get_training_optimizers import *


PATHS = Paths(encoder=ENCODER_MODEL)
print(ENCODER_MODEL)

ENCODER = Encoders(model=ENCODER_MODEL,
                   checkpoint_path='../../' + PATHS._load_encoder_path(encoder_loader=ENCODER_LOADER, augment=True),
                   device=DEVICE)



image_model, dim = ENCODER._get_encoder_model()

features_list  = pickle.load(open('../../' +PATHS._get_features_path('TRAIN'), 'rb'))
paths_list  = pickle.load(open('/home/starksultana/Documentos/MEIC/5o_ano/Tese/code/remote-sensing-image-captioning/src/encoder_scripts/cnn_alt_image_paths.pickle', 'rb'))

print(features_list[0].shape)
print(len(paths_list))
#
def save_index():
    faiss_index = faiss.IndexFlatL2(2048)
    for feature in features_list:
    # features = np.vstack(features_list)
        for feat in feature:
            # feat = feat.view(feat.size()[0], -1, feat.size()[-1])
            # print(feat.shape)
            feat = feat.flatten(start_dim =0, end_dim =1).mean(dim=0)
    # print(descriptors.shape)
            faiss_index.add(np.array(feat.unsqueeze(0)))
    #
    faiss.write_index(faiss_index, 'index_alt')

def search(image):
    data_transform = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
                      transforms.RandomVerticalFlip(), transforms.RandomRotation(90),
                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])])

    faiss_index = faiss.read_index('index_alt')

    query_image = image
    img = cv2.imread(image)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)

    img = cv2.resize(img, (224, 224))
    img = img.transpose(2, 0, 1)

    assert img.shape == (3, 224, 224)
    assert np.max(img) <= 255
    # #
    img = torch.FloatTensor(img / 255.)
    input_tensor = data_transform(img)
    input_tensor = input_tensor.view(1, *input_tensor.shape)
    # print(input_tensor.shape)

    with torch.no_grad():
        query_descriptors = image_model.extract_features(input_tensor.to(DEVICE))
    #     # print(query_descriptors)
    features = query_descriptors.permute(0, 2, 3, 1).flatten(start_dim=0, end_dim=2).mean(dim=0).cpu()
    input = np.array(features.unsqueeze(0)) #features_list[1][30].flatten(start_dim=0, end_dim=1).mean(dim=0).cpu()
    # # print(input.shape)
    distance, indices = faiss_index.search(input, k = 9)
    #
    print(indices)
    print(distance,indices)
    fig, ax = plt.subplots(3, 3, figsize=(15,15))
    for file_index, ax_i in zip(indices[0], np.array(ax).flatten()):
        ax_i.imshow(plt.imread('/home/starksultana/Documentos/MEIC/5o_ano/Tese/code/remote-sensing-image-captioning/data/images/RSICD_images/' + paths_list[file_index]))

    plt.show()
# #
save_index()
#
search('/home/starksultana/Documentos/MEIC/5o_ano/Tese/code/remote-sensing-image-captioning/data/images/RSICD_images/airport_100.jpg')
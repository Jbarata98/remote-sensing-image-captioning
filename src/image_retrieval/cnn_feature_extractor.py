from torchvision import transforms
from tqdm import tqdm

# import sys #for colab
# sys.path.insert(0,'/content/gdrive/MyDrive/Tese/code')

from src.configs.get_models import *
from src.configs.get_data_paths import *
from src.configs.datasets import FeaturesDataset

from src.configs.get_training_optimizers import *
from matplotlib import pyplot

import os

PATHS = Paths(encoder=ENCODER_MODEL)
print(ENCODER_MODEL)

ENCODER = Encoders(model=ENCODER_MODEL,
                   checkpoint_path='../' + PATHS._load_encoder_path(encoder_loader=ENCODER_LOADER, augment=True), device=DEVICE)

data_folder = PATHS._get_input_path(is_classification=True)
data_name = DATASET + '_CLASSIFICATION_dataset'

# make sure path exists before running all the code
if os.path.exists('../' + PATHS._get_features_path('TRAIN')):
    print('path_exists')


# function to visualize the feature maps
def visualize_fmap(features):
    square = 8
    for fmap in features:
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


class extract_features():

    def __init__(self, device):

        self.device = device
        self.image_model, self.dim = ENCODER._get_encoder_model()

        if ENCODER_MODEL == ENCODERS.RESNET.value:
            modules = list(self.image_model.children())[:-2]
            self.image_model = nn.Sequential(*modules)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

        for p in self.image_model.parameters():
            p.requires_grad = False

    def _extract(self, images):
        if ENCODER_MODEL == ENCODERS.EFFICIENT_NET_IMAGENET_FINETUNED.value or ENCODER_MODEL == ENCODERS.EFFICIENT_NET_IMAGENET.value:

            out = self.image_model.extract_features(images)

            #use resnet
        else:
            out = self.image_model(images)

        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)

        return out


if __name__ == "__main__":


    data_transform = [transforms.RandomResizedCrop(256), transforms.RandomHorizontalFlip(),
                      transforms.RandomVerticalFlip(), transforms.RandomRotation(90),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                            std=[0.229, 0.224, 0.225])]
    f_extractor = extract_features(DEVICE)

    split = 'TRAIN'

    print("split:", split)

    features = []
    imgs = torch.utils.data.DataLoader(
        FeaturesDataset(data_folder, data_name, split, transform=transforms.Compose(data_transform)), batch_size=32,
        shuffle=False, num_workers=1, pin_memory=True)

    # f_tensor = torch.zeros(len(imgs),7,7,2048).to(DEVICE)

    with tqdm(total=len(imgs)) as pbar:

        for i, img in enumerate(imgs):
            fmap = f_extractor._extract(img)

            features.append(fmap)
            # f_tensor[i] = fmap
            pbar.update(1)

    # dump the features into pickle file
    pickle.dump(features, open('../' + PATHS._get_features_path(split), 'wb'))

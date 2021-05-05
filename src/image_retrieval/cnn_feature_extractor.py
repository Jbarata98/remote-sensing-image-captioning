from torchvision import transforms
from tqdm import tqdm

# import sys #for colab
# sys.path.insert(0,'/content/gdrive/MyDrive/Tese/code')

from src.configs.getters.get_models import *
from src.configs.getters.get_data_paths import *
from src.configs.utils.datasets import FeaturesDataset

from src.configs.getters.get_training_optimizers import *

import os

PATHS = Paths(encoder=ENCODER_MODEL)
print("using {} as the encoder".format(ENCODER_MODEL))

# define encoder to extract features
ENCODER = Encoders(model=ENCODER_MODEL,
                   checkpoint_path='../' + PATHS._load_encoder_path(encoder_name=ENCODER_LOADER), device=DEVICE)

# define input folders and general data name
data_folder = PATHS._get_input_path(is_classification=True)
data_name = DATASET + '_CLASSIFICATION_dataset'

# make sure path exists before running all the code
if os.path.exists('../' + PATHS._get_features_path('TRAIN')):
    print('feature extracting path exists')


class ExtractFeatures:
    """
    class to extract the feature maps
    """

    def __init__(self, device):

        self.device = device
        self.image_model, self.dim = ENCODER._get_encoder_model()

        # if using a resnet cannot use .extract_features method
        if ENCODER_MODEL == ENCODERS.RESNET.value:
            modules = list(self.image_model.children())[:-2]
            self.image_model = nn.Sequential(*modules)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

        for p in self.image_model.parameters():
            p.requires_grad = False

    def _extract(self, images):

        # use resnet
        if ENCODERS.RESNET.value:

            out = self.image_model(images)

        else:
            out = self.image_model.extract_features(images)

        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)

        return out


if __name__ == "__main__":

    data_transform = [transforms.RandomHorizontalFlip(),
                      transforms.RandomVerticalFlip(), transforms.RandomAffine([90, 180, 270]),
                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])]
    f_extractor = ExtractFeatures(DEVICE)

    split = 'TRAIN'
    b_size = 32

    features = []
    imgs = torch.utils.data.DataLoader(
        FeaturesDataset(data_folder, data_name, split, transform=transforms.Compose(data_transform)), batch_size=b_size,
        shuffle=False, num_workers=1, pin_memory=True)

    # f_tensor = torch.zeros(len(imgs),7,7,2048).to(DEVICE)

    with tqdm(total=len(imgs)) as pbar:

        for img in imgs:
            fmap = f_extractor._extract(img)

            features.append(fmap)

            pbar.update(1)

    # dump the features into pickle file
    pickle.dump(features, open('../' + PATHS._get_features_path(split), 'wb'))

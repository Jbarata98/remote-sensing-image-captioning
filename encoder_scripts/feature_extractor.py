from configs.get_models import *
from configs.globals import *
from configs.get_data_paths import *
from configs.datasets import FeaturesDataset

from encoder_scripts.create_classification_data import create_classes_json,create_classification_files
from configs.get_training_optimizers import *

import tqdm

PATHS = Paths(model=ENCODER_MODEL)

ENCODER = Encoders(model=ENCODER_MODEL, checkpoint_path=PATHS._load_encoder_path(encoder_loader=ENCODER_LOADER),device = DEVICE)

data_folder = PATHS._get_input_path(is_classification=True)
data_name = DATASET + '_CLASSIFICATION_dataset'

class extract_features():

    def __init__(self, device):

        self.device = device
        self.image_model, self.dim = ENCODER._get_encoder_model()
        #encoded image size will equal 14
        self.adaptive_pool = nn.AdaptiveAvgPool2d((14, 14))

    def _extract(self, images):


        out = self.image_model.extract_features(images)

        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)


        return out


if __name__ == "__main__":

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(FeaturesDataset(data_folder, data_name, 'TRAIN', transform=transforms.Compose([normalize])), batch_size = 8, shuffle=True, num_workers=1, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(FeaturesDataset(data_folder, data_name, 'VAL', transform=transforms.Compose([normalize])), batch_size = 8, shuffle = False, num_workers = 1,pin_memory = True)
    test_loader = torch.utils.data.DataLoader(FeaturesDataset(data_folder, data_name, 'TEST', transform=transforms.Compose([normalize])), batch_size = 8, shuffle = False, num_workers = 1, pin_memory = True),

    f_extractor = extract_features(DEVICE)
    features = []
    for i,imgs in tqdm.tqdm(enumerate(train_loader), desc = "Loading Imgs"):
        fmap = f_extractor._extract(imgs)
        features.append(fmap)

    print(len(features))






from configs.get_models import *
from configs.globals import *
from configs.get_data_paths import *
from configs.datasets import FeaturesDataset

from encoder_scripts.create_classification_data import create_classes_json,create_classification_files
from configs.get_training_optimizers import *
from matplotlib import pyplot
import tqdm
import pickle

PATHS = Paths(model=ENCODER_MODEL)

ENCODER = Encoders(model=ENCODER_MODEL, checkpoint_path=PATHS._load_encoder_path(encoder_loader=ENCODER_LOADER),device = DEVICE)

data_folder = PATHS._get_input_path(is_classification=True)
data_name = DATASET + '_CLASSIFICATION_dataset'

class extract_features():

    def __init__(self, device):

        self.device = device
        self.image_model, self.dim = ENCODER._get_encoder_model()
        #encoded image size will equal 14
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

        for p in self.image_model.parameters():
            p.requires_grad = False

    def _extract(self, images):

        out = self.image_model.extract_features(images)

        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)

        return out

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

if __name__ == "__main__":

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    f_extractor = extract_features(DEVICE)
    features = []

    for split in ('TRAIN', 'VAL', 'TEST'):

        print("split:", split)

        imgs = torch.utils.data.DataLoader(
            FeaturesDataset(data_folder, data_name, split, transform=transforms.Compose([normalize])), batch_size=1,
            shuffle=False, num_workers=1, pin_memory=True)

        with tqdm.tqdm(total=len(imgs)) as pbar:
            for i,img in enumerate(imgs):

                fmap = f_extractor._extract(img)

                features.append(fmap)
                pbar.update(1)



        pickle.dump(features, open('experiments/encoder/' + DATASET + '_features_' + split + '.pickle', 'wb'))






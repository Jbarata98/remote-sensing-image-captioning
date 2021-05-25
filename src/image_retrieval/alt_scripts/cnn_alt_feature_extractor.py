import json

from torchvision import transforms
from tqdm import tqdm
# import sys #for colab
# sys.path.insert(0,'/content/gdrive/MyDrive/Tese/code')

from src.configs.utils.datasets import FeaturesDataset

from src.configs.setters.set_initializers import *

data_folder = PATHS._get_input_path(is_classification=True)
data_name = DATASET + '_CLASSIFICATION_dataset'

PATHS = Paths(encoder=ENCODER_MODEL)
print(ENCODER_MODEL)

ENCODER = Encoders(model=ENCODER_MODEL,
                   checkpoint_path='../' + PATHS._get_pretrained_encoder_path(encoder_loader=ENCODER_LOADER),
                   device=DEVICE)

data_transform = [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
                  transforms.RandomVerticalFlip(), transforms.RandomRotation(90),
                  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])]

imgs = torch.utils.data.DataLoader(
    FeaturesDataset(data_folder, data_name, 'TRAIN', transform=transforms.Compose(data_transform)), batch_size=1,
    shuffle=False, num_workers=1, pin_memory=True)


image_model, dim = ENCODER._get_encoder_model()

if __name__ == "__main__":

    def get_image_name(dataset='remote_sensing'):
        # get captions path to retrieve image name
        train_filenames = []

        if dataset == 'remote_sensing':
            file = open('../' + PATHS._get_captions_path())
            data = json.load(file)
            for image in data['images']:
                if image['split'] == "train":
                    train_filenames.append(image['filename'])
        return train_filenames

    descriptors = []
    image_paths = []
    image_model.to(DEVICE)
    train_filenames = get_image_name()
    # print(train_filenames)
    # print(len(train_filenames))
    with torch.no_grad():
        image_model.eval()
        for path,img in zip(train_filenames,tqdm(imgs)):
            result = image_model.Extract_features(img.to(DEVICE))
            results = result.permute(0,2,3,1).flatten(start_dim =0, end_dim =2).mean(dim=0)
            descriptors.append(results.cpu().numpy())
            image_paths.append(path)

    pickle.dump(image_paths, open('cnn_alt_image_paths.pickle', 'wb'))
    pickle.dump(descriptors, open('../../../experiments/encoder/features/cnn_alt_features.pickle', 'wb'))

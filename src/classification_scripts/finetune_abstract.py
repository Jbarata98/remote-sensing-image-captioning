from torchvision.transforms import transforms

from src.classification_scripts.augment import TwoViewTransform, CustomRotationTransform
from src.configs.setters.set_initializers import *
import json
import h5py

# fine tune always true when running this script
from src.configs.utils.datasets import ClassificationDataset

FINE_TUNE = True

setters_class = Setters(file="classification_scripts/encoder_training_details.txt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# training parameters
h_parameters = setters_class._set_training_parameters()
PATHS = setters_class._set_paths()

# set encoder
ENCODER = setters_class._set_encoder(
    path='../' + PATHS._get_pretrained_encoder_path(
        encoder_name=ENCODER_LOADER))

# set optimizers
OPTIMIZERS = setters_class._set_optimizer()

# set data names
data_folder = PATHS._get_input_path(is_classification=True)
data_name = DATASET + '_CLASSIFICATION_dataset'

DEBUG = False

logging.basicConfig(
    format='%(levelname)s: %(message)s', level=logging.INFO)


class FineTune:
    """
    class that unfreezes the efficient-net model and pre-trains it on RSICD data
    """

    def __init__(self, model_type, device, nr_classes=31,
                 enable_finetuning=FINE_TUNE):  # default is 31 classes (nr of rscid classes)
        self.device = device

        logging.info("Running encoder fine-tuning script...")

        self.model_type = model_type
        self.classes = nr_classes
        self.enable_finetuning = enable_finetuning
        self.device = device
        self.checkpoint_exists = False

        image_model, dim = ENCODER._get_encoder_model()

        self.model = image_model.to(self.device)

    def _setup_train(self):
        optimizer = OPTIMIZERS._get_optimizer(
            params=filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=float(h_parameters['encoder_lr'])) if self.enable_finetuning else None

        self.optimizer = optimizer
        self.criterion = OPTIMIZERS._get_loss_function()
        self._load_weights_from_checkpoint(load_to_train=True)

        return self.model

    def _setup_transforms(self):

        with open(os.path.join(PATHS._get_input_path(is_classification=True), 'DICT_LABELS_' + '.json'), 'r') as j:
            classes = json.load(j)

        print("h_parameters:", h_parameters)
        print("nr of classes:", len(classes))

        # load target images for histogram matching if dealing with training data
        target_h = h5py.File(os.path.join(data_folder, 'TEST_IMAGES_' + data_name + '.hdf5'), 'r')
        self.target_imgs = target_h['images']

        self.data_transform = [transforms.RandomHorizontalFlip(),
                          transforms.RandomVerticalFlip(),
                          CustomRotationTransform(angles=[90, 180, 270]),
                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])]

    def _setup_dataloaders(self):

        # loaders
        self.train_loader = torch.utils.data.DataLoader(
            ClassificationDataset(data_folder, data_name, 'TRAIN',
                                  transform=TwoViewTransform(transforms.Compose(self.data_transform), self.target_imgs) if LOSS == LOSSES.SupConLoss.value
                                  else transforms.Compose(self.data_transform)),
            batch_size=int(h_parameters['batch_size']), shuffle=True, num_workers=int(h_parameters['workers']),
            pin_memory=True)

        self.val_loader = torch.utils.data.DataLoader(
            ClassificationDataset(data_folder, data_name, 'VAL',
                                  transform=transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                     std=[0.229, 0.224, 0.225])])),
            batch_size=int(h_parameters['batch_size']), shuffle=False, num_workers=int(h_parameters['workers']),
            pin_memory=True)

    def _load_weights_from_checkpoint(self, load_to_train):

        if os.path.exists('../../' + PATHS._get_pretrained_encoder_path(encoder_name=ENCODER_LOADER)):
            logging.info("checkpoint exists, loading...")
            if torch.cuda.is_available():
                checkpoint = torch.load('../../' + PATHS._get_pretrained_encoder_path(encoder_name=ENCODER_LOADER))
            else:
                checkpoint = torch.load('../../' + PATHS._get_pretrained_encoder_path(encoder_name=ENCODER_LOADER),
                                        map_location=torch.device("cpu"))

            self.checkpoint_exists = True

            # load model weights
            self.model.load_state_dict(checkpoint['model'])

            if load_to_train:
                # load optimizers and start epoch
                self.checkpoint_start_epoch = checkpoint['epoch'] + 1
                self.checkpoint_epochs_since_last_improvement = checkpoint['epochs_since_improvement']
                self.checkpoint_val_loss = checkpoint['val_loss']
                # load weights for encoder

                self.optimizer.load_state_dict(checkpoint['optimizer'])

                logging.info(
                    "Restore model from checkpoint. Start epoch %s ", self.checkpoint_start_epoch)
        else:
            logging.info(
                "No checkpoint. Will start model from beggining\n")


    # Checkpoint saver
    def _save_checkpoint_encoder(self, val_loss_improved, epoch, epochs_since_improvement, val_loss
                                 ):
        if val_loss_improved:
            state = {'epoch': epoch,
                     'epochs_since_improvement': epochs_since_improvement,
                     'val_loss': val_loss,
                     'model': self.model.state_dict(),
                     'optimizer': self.optimizer.state_dict()
                     }

            filename_checkpoint = '../../' + PATHS._get_pretrained_encoder_path(encoder_name=ENCODER_LOADER)
            torch.save(state, filename_checkpoint)
            # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint

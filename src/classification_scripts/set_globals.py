from src.configs.setters.set_initializers import *


def _set_globals(file = 'classification_scripts/encoder_training_details.txt'):
    # print(sys.path)

    setters_class = Setters(file= file)

    FINE_TUNE = True

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


    return {"FINE_TUNE" : FINE_TUNE,
            "DEVICE" : DEVICE,
            "h_parameters":h_parameters,
            "PATHS": PATHS,
            "ENCODER": ENCODER,
            "OPTIMIZERS": OPTIMIZERS,
            "data_folder": data_folder,
            "data_name": data_name,
            "DEBUG": DEBUG}
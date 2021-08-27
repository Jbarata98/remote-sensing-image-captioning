from src.classification_scripts.set_classification_globals import _set_globals

setters = _set_globals('../../configs/setters/training_details.txt')

encoder  = setters["ENCODER"]._get_encoder_model()
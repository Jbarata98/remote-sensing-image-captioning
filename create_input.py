from configs.utils import *
from configs.enums_file import *
from configs.paths_generator import *

ARCHITECTURE = ARCHITECTURES.BASELINE

if __name__ == '__main__':
    # Create input files (along with word map)

    create_input_files(dataset=DATASETS.RSICD,
                       json_path='captions/dataset_rsicd_modified.json',  # path of the .json file with the captions
                       image_folder='images/RSICD_images',  # folder containing the images
                       captions_per_image=5,
                       min_word_freq=2,
                       output_folder=PATH_DATA(ARCHITECTURES.BASELINE, input=True, fine_tune=False),
                       max_len=30)

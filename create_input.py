from configs.utils import *


if __name__ == '__main__':
    # Create input files (along with word map)

    create_input_files(dataset = DATASET,
                       json_path='captions/dataset_rsicd_modified.json',  # path of the .json file with the captions
                       image_folder='images/RSICD_images',  # folder containing the images
                       captions_per_image=5,
                       min_word_freq=2,
                       output_folder=PATH_DATA(ARCHITECTURE, input=True, fine_tune=fine_tune_encoder),
                       max_len=30)

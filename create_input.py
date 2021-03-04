from utils import *


if __name__ == '__main__':
    # Create input files (along with word map)
    create_input_files(dataset='rsicd',
                       json_path='caption_datasets/dataset_rsicd_modified.json',
                       image_folder='images/RSICD_images',
                       captions_per_image=5,
                       min_word_freq=2,
                       output_folder= ARCHITECTURE + '/inputs/',
                       max_len=30)

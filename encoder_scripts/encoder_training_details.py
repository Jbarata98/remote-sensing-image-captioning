import torch.backends.cudnn as cudnn
from configs.get_data_paths import *

# Fine_tune
fine_tune = True  # fine-tune encoder

checkpoint_encoder_path = get_path(model = ENCODER_MODEL,best_checkpoint = True, is_encoder = True, fine_tune = True)

# Model parameters
dropout = 0.5
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Training parameters
start_epoch = 0
epochs = 50  # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
batch_size = 8
workers = 1  # for data-loading; right now, only 1 works with h5py
encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
grad_clip = 5.  # clip gradients at an absolute value of
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
best_bleu4 = 0.  # BLEU-4 score right now
print_freq = 100  # print training/validation stats every __ batches

# Debug
DEBUG = False
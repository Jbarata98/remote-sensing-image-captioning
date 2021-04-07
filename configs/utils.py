from configs.get_models import *
from configs.get_data_paths import *
from configs.get_training_optimizers import *
from configs.get_training_details import *
from configs.datasets import *
from configs.get_training_optimizers import *

# Initializers

# set hyperparameters

HPARAMETER = Training_details("configs/training_details.txt")
h_parameter = HPARAMETER._get_training_details()

# parameters for main filename
caps_per_img = int(h_parameter['captions_per_image'])
min_word_freq = int(h_parameter['min_word_freq'])

# base name shared by data files {nr of captions per img and min word freq in create_input_files.py}
data_name = DATASET + "_" + str(caps_per_img) + "_cap_per_img_" + str(
    min_word_freq) + "_min_word_freq"  # DATASET + '_CLASSIFICATION_dataset'
figure_name = DATASET + "_" + ENCODER_MODEL + "_" + ATTENTION  # when running visualization

# set paths
PATHS = Paths(architecture=ARCHITECTURE, attention=ATTENTION, model=ENCODER_MODEL, filename=data_name,
              figure_name=figure_name, dataset=DATASET, fine_tune=FINE_TUNE)
# set encoder
ENCODER = Encoders(model=ENCODER_MODEL, checkpoint_path=PATHS._load_encoder_path(encoder_loader=ENCODER_LOADER),
                   device=DEVICE)
# set AuxLM
AuxLM = AuxLM(model=AUX_LM, device=DEVICE)

transf_tokenizer, transf_model = AuxLM._get_decoder_model(special_tokens=SPECIAL_TOKENS)
# set optimizers
OPTIMIZER = Optimizers(optimizer_type=OPTIMIZER, loss_func=LOSS, device=DEVICE)

data_folder = PATHS._get_input_path()  # folder with data files saved by create_input_files.py
checkpoint_model = PATHS._get_checkpoint_path()  # get_path(ARCHITECTURE, model = MODEL, data_name=data_name,checkpoint = True, best_checkpoint = True, fine_tune = fine_tune_encoder) #uncomment for checkpoint

# name of wordmap
word_map_file = data_folder + 'WORDMAP_' + data_name + '.json'  # word map, ensure it's the same the data was encoded with and the model was trained with


def init_embedding(embeddings):
    """
    Fills embedding tensor with values from the uniform distribution.
    :param embeddings: embedding tensor
    """
    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)


def load_embeddings(emb_file, word_map):
    """
    Creates an embedding tensor for the specified word map, for loading into the model.
    :param emb_file: file containing embeddings (stored in GloVe format)
    :param word_map: word map
    :return: embeddings in the same order as the words in the word map, dimension of embeddings
    """

    # Find embedding dimension
    with open(emb_file, 'r') as f:
        emb_dim = len(f.readline().split(' ')) - 1

    vocab = set(word_map.keys())

    # Create tensor to hold embeddings, initialize
    embeddings = torch.FloatTensor(len(vocab), emb_dim)
    init_embedding(embeddings)

    # Read embedding file
    print("\nLoading embeddings...")
    for line in open(emb_file, 'r'):
        line = line.split(' ')

        emb_word = line[0]
        embedding = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))

        # Ignore word if not in train_vocab
        if emb_word not in vocab:
            continue

        embeddings[word_map[emb_word]] = torch.FloatTensor(embedding)

    return embeddings, emb_dim


def save_checkpoint(epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer, decoder_optimizer,
                    bleu4, is_best):
    """
    Saves model checkpoint.
    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    :param bleu4: validation BLEU-4 score for this epoch
    :param is_best: is this checkpoint the best so far?
    """
    if is_best:
        state = {'epoch': epoch,
                 'epochs_since_improvement': epochs_since_improvement,
                 'bleu-4': bleu4,
                 'encoder': encoder,
                 'decoder': decoder,
                 'encoder_optimizer': encoder_optimizer,
                 'decoder_optimizer': decoder_optimizer}

        filename_best_checkpoint = Paths._get_checkpoint_path()
        torch.save(state, filename_best_checkpoint)

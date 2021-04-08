import torch
import numpy as np

class create_embeddings():

    def __init__(self, emb_tensor, emb_file, emb_vocab):

        self.emb_tensor = emb_tensor
        self.emb_file = emb_file
        self.emb_vocab = emb_vocab

    def _init_embeddings(self):

        """
        Fills embedding tensor with values from the uniform distribution.
        :param embeddings: embedding tensor
        """

        bias = np.sqrt(3.0 / self.emb_tensor.size(1))
        torch.nn.init.uniform_(self.emb_tensor, -bias, bias)

    def _load_embeddings(self):

        """
        Creates an embedding tensor for the specified word map, for loading into the model.
        :param self.emb_file: file containing embeddings (stored in GloVe format)
        :param word_map: word map
        :return: embeddings in the same order as the words in the word map, dimension of embeddings
        """
    
        # Find embedding dimension
        with open(self.emb_file, 'r') as f:
            emb_dim = len(f.readline().split(' ')) - 1
    
        vocab = set(self.emb_vocab.keys())
    
        # Create tensor to hold embeddings, initialize
        embeddings = torch.FloatTensor(len(vocab), emb_dim)
        self._init_embeddings()
    
        # Read embedding file
        print("\nLoading embeddings...")
        for line in open(self.emb_file, 'r'):
            line = line.split(' ')
    
            emb_word = line[0]
            embedding = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))
    
            # Ignore word if not in train_vocab
            if emb_word not in vocab:
                continue
    
            embeddings[self.emb_vocab[emb_word]] = torch.FloatTensor(embedding)
    
        return embeddings, emb_dim

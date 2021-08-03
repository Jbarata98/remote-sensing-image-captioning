"""Script for fine-tuning Pegasus

  #Script to fine-tune Pegasus for captioning task
  - Similar images captions as input,
  - target is a random sentence from the 5


  # use Pegasus xsum model as base for fine-tuning
  model_name = 'google/pegasus-large'
  train_dataset, _, _, tokenizer = prepare_data(model_name, train_texts, train_labels)
  trainer = prepare_fine_tuning(model_name, tokenizer, train_dataset)
  trainer.train()

Reference:
  https://gist.github.com/jiahao87/50cec29725824da7ff6dd9314b53c4b3
"""
import collections
import random as rand

import torch
import json
from transformers import PegasusForConditionalGeneration, PegasusTokenizer, Trainer, TrainingArguments
from src.configs.setters.set_initializers import *


setters = Setters(file='../../../configs/setters/training_details.txt')

paths = setters._set_paths()

class PegasusFinetuneDataset(torch.utils.data.Dataset):
    """
    Pegasus Dataset specific for fine-tuning task
    """
    def __init__(self, encodings, targets):
        self.encodings = encodings
        self.target = targets

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels['input_ids'][idx])  # torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels['input_ids'])  # len(self.labels)

def get_data(filename, save_file = False):
    """
    gets raw data ( captions )
    extracts label (target sentence is a random one from the 5)
    returns dictionary for data and target
    """
    captions_split = collections.defaultdict(dict)
    train_captions = collections.defaultdict(list)
    val_captions = collections.defaultdict(list)
    test_captions = collections.defaultdict(list)
    with open('../../../' + filename, 'r') as paths_file:
        caption_dataset = json.load(paths_file)
    for img_id in caption_dataset['images']:
        for sentence in img_id['sentences']:
            if img_id['split'] == 'train':
                train_captions[img_id['filename']].append(sentence['raw'])
                captions_split['train'].update(train_captions)
            elif img_id['split'] == 'val':
                val_captions[img_id['filename']].append(sentence['raw'])
                captions_split['val'].update(val_captions)
            elif img_id['split'] == 'test':
                test_captions[img_id['filename']].append(sentence['raw'])
                captions_split['test'].update(test_captions)

    target_dict = collections.defaultdict(dict)
    train_target_captions = collections.defaultdict(list)
    val_target_captions = collections.defaultdict(list)
    test_target_captions = collections.defaultdict(list)
    for split in captions_split:
        for filename in captions_split[split]:
            if split == "train":
                train_target_captions[filename].append(captions_split[split][filename][rand.randint(0, 4)])
                target_dict[split].update(train_target_captions)
            if split == "val":
               val_target_captions[filename].append(captions_split[split][filename][rand.randint(0, 4)])
               target_dict[split].update(val_target_captions)
            if split == "test":
                test_target_captions[filename].append(captions_split[split][filename][rand.randint(0, 4)])
                target_dict[split].update(test_target_captions)

    if save_file:
        with open('../../../' + paths._get_input_path() + 'raw_captions_dataset', 'w') as raw_dataset:
            logging.info("dumped raw captions...")
            json.dump(captions_split,raw_dataset)
        with open('../../../' + paths._get_input_path() + 'target_captions_dataset', 'w') as target_dataset:
            logging.info("dumped target raw captions...")
            json.dump(target_dict, target_dataset)



def prepare_data(model_name):
    """
    Prepare input data for model fine-tuning
    """

    if not os.path.exists('../../../' + paths._get_input_path() + 'target_captions_dataset'):
        print("Pre-train dataset doesn't exist..")
        get_data(paths._get_captions_path(), save_file=False)
    else:
        print("Pre-train dataset already exists..")

    with open('../../../' + paths._get_input_path() + 'target_captions_dataset', 'r') as captions_file:
        captions_dataset = json.load(captions_file)

    with open('../../../' + paths._get_input_path() + 'target_captions_dataset', 'r') as target_file:
        target_dataset = json.load(target_file)

    train_dict, target_train_dict = captions_dataset["train"], target_dataset["train"]
    val_dict, target_val_dict = captions_dataset["val"], target_dataset["val"]
    test_dict, target_test_dict = captions_dataset["test"], target_dataset["test"]

    # tokenizer = PegasusTokenizer.from_pretrained(model_name)

    # def tokenize_data(texts, labels):
    #     """
    #     tokenizes the data for pegasus input
    #     """
    #     encodings = tokenizer(texts, truncation=True, padding='max_length')
    #     decodings = tokenizer(labels, truncation=True, padding='max_length')
    #     dataset_tokenized = PegasusFinetuneDataset(encodings, decodings)
    #     return dataset_tokenized
    #
    # train_dataset = tokenize_data(train_texts, train_labels)
    # val_dataset = tokenize_data(val_texts, val_labels)
    # test_dataset = tokenize_data(test_texts, test_labels)

    # return train_dataset, val_dataset, test_dataset, tokenizer

#def compute_metrics
#computes bleu-4

def prepare_fine_tuning(model_name, tokenizer, train_dataset, val_dataset=None, freeze_encoder=False, output_dir='./results'):
  """
  Prepare configurations and base model for fine-tuning
  """
  torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
  model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

  if freeze_encoder:
    for param in model.model.encoder.parameters():
      param.requires_grad = False

  if val_dataset is not None:
    training_args = TrainingArguments(
      output_dir=output_dir,           # output directory
      num_train_epochs=2000,           # total number of training epochs
      per_device_train_batch_size=1,   # batch size per device during training, can increase if memory allows
      per_device_eval_batch_size=1,    # batch size for evaluation, can increase if memory allows
      save_steps=500,                  # number of updates steps before checkpoint saves
      save_total_limit=5,              # limit the total amount of checkpoints and deletes the older checkpoints
      evaluation_strategy='steps',     # evaluation strategy to adopt during training
      eval_steps=100,                  # number of update steps before evaluation
      warmup_steps=500,                # number of warmup steps for learning rate scheduler
      weight_decay=0.01,               # strength of weight decay
      logging_dir='./logs',            # directory for storing logs
      logging_steps=10,
    )

    trainer = Trainer(
      model=model,                         # the instantiated 🤗 Transformers model to be trained
      args=training_args,                  # training arguments, defined above
      train_dataset=train_dataset,         # training dataset
      eval_dataset=val_dataset,            # evaluation dataset
      tokenizer=tokenizer
    )

  else:
    training_args = TrainingArguments(
      output_dir=output_dir,           # output directory
      num_train_epochs=2000,           # total number of training epochs
      per_device_train_batch_size=1,   # batch size per device during training, can increase if memory allows
      save_steps=500,                  # number of updates steps before checkpoint saves
      save_total_limit=5,              # limit the total amount of checkpoints and deletes the older checkpoints
      warmup_steps=500,                # number of warmup steps for learning rate scheduler
      weight_decay=0.01,               # strength of weight decay
      logging_dir='./logs',            # directory for storing logs
      logging_steps=10,
    )

    trainer = Trainer(
      model=model,                         # the instantiated 🤗 Transformers model to be trained
      args=training_args,                  # training arguments, defined above
      train_dataset=train_dataset,         # training dataset
      tokenizer=tokenizer
    )

  return trainer


# if __name__ == '__main__':
#     # use XSum dataset as example, with first 1000 docs as training data
#     from datasets import load_dataset
#
#     dataset = load_dataset("xsum")
#     train_texts, train_labels = dataset['train']['document'][:1000], dataset['train']['summary'][:1000]
#
#     # use Pegasus Large model as base for fine-tuning
#     model_name = 'google/pegasus-large'
#     train_dataset, _, _, tokenizer = prepare_data(model_name, train_texts, train_labels)
#     trainer = prepare_fine_tuning(model_name, tokenizer, train_dataset)
#     trainer.train()

prepare_data("lol")


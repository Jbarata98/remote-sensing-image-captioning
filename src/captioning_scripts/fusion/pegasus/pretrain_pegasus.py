"""Script for fine-tuning Pegasus"""
# import sys
# sys.path.insert(0, '/content/gdrive/MyDrive/Tese/code')  # for colab
import random
import random as rand

import json

import numpy as np
import transformers
from datasets import load_metric
from nltk.translate.bleu_score import corpus_bleu
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from src.configs.setters.set_initializers import *

setters = Setters(file='../../../configs/setters/training_details.txt')

paths = setters._set_paths()


class PegasusFinetuneDataset(torch.utils.data.Dataset):
    """
    Pegasus Dataset specific for fine-tuning task
    """

    def __init__(self, encodings, targets):
        self.encodings = encodings
        self.targets = targets

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.targets['input_ids'][idx])  # torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.targets['input_ids'])  # len(self.labels)


def get_data(filename, save_file=False):
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
                if len(train_captions[img_id['filename']]) == int(
                        setters._set_training_parameters()["captions_per_image"]):  # means its full
                    # shuffles the captions
                    random.shuffle(train_captions[img_id['filename']])
                captions_split['train'].update(train_captions)
            elif img_id['split'] == 'val':
                val_captions[img_id['filename']].append(sentence['raw'])
                if len(val_captions[img_id['filename']]) == int(setters._set_training_parameters()[
                                                                    "captions_per_image"]):  # means its full                  random.shuffle(val_captions)
                    # shuffles the captions
                    random.shuffle(val_captions[img_id['filename']])
                captions_split['val'].update(val_captions)
            elif img_id['split'] == 'test':
                test_captions[img_id['filename']].append(sentence['raw'])
                if len(test_captions[img_id['filename']]) == int(
                        setters._set_training_parameters()["captions_per_image"]):  # means its full
                    # shuffles the captions
                    random.shuffle(test_captions[img_id['filename']])
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
            json.dump(captions_split, raw_dataset)
        with open('../../../' + paths._get_input_path() + 'target_captions_dataset', 'w') as target_dataset:
            logging.info("dumped target raw captions...")
            json.dump(target_dict, target_dataset)


def prepare_data():
    """
    Prepare input data for model fine-tuning
    """

    if not os.path.exists('../../../' + paths._get_input_path() + 'raw_captions_dataset'):
        logging.info("Pre-train dataset doesn't exist..")
        get_data(paths._get_captions_path(), save_file=True)
    else:
        logging.info("Pre-train dataset already exists..")

    with open('../../../' + paths._get_input_path() + 'raw_captions_dataset', 'r') as captions_file:
        captions_dataset = json.load(captions_file)

    with open('../../../' + paths._get_input_path() + 'target_captions_dataset', 'r') as target_file:
        target_dataset = json.load(target_file)

    with open('../../../../' + paths._get_similarity_mapping_path(nr_similarities=1), 'r') as hashmap_file:
        hashmap = json.load(hashmap_file)

    train_dict, target_train_dict = captions_dataset["train"], target_dataset["train"]
    val_dict, target_val_dict = captions_dataset["val"], target_dataset["val"]
    test_dict, target_test_dict = captions_dataset["test"], target_dataset["test"]

    train_texts = [' '.join(train_dict.get(hashmap.get(img_name)['Most similar'])) for img_name in train_dict.keys()]
    train_labels = [' '.join(target_train_dict.get(hashmap.get(img_name)['Most similar'])) for img_name in
                    target_train_dict.keys()]
    val_texts = [' '.join(train_dict.get(hashmap.get(img_name)['Most similar'])) for img_name in val_dict.keys()]
    val_labels = [' '.join(target_train_dict.get(hashmap.get(img_name)['Most similar'])) for img_name in
                  target_val_dict.keys()]
    test_texts = [' '.join(train_dict.get(hashmap.get(img_name)['Most similar'])) for img_name in test_dict.keys()]
    test_labels = [' '.join(target_train_dict.get(hashmap.get(img_name)['Most similar'])) for img_name in
                   target_test_dict.keys()]

    assert len(train_texts) == len(train_labels)
    assert len(val_texts) == len(val_labels)
    assert len(test_texts) == len(test_labels)

    AuxLM = setters._set_aux_lm()

    def tokenize_data(texts, labels):
        """
        tokenizes the data for pegasus input
        """

        encodings = AuxLM["tokenizer"](texts, truncation=True, padding='longest')
        decodings = AuxLM["tokenizer"](labels, truncation=True, padding='longest')
        dataset_tokenized = PegasusFinetuneDataset(encodings, decodings)
        return dataset_tokenized

    #
    train_dataset = tokenize_data(train_texts, train_labels)
    val_dataset = tokenize_data(val_texts, val_labels)
    test_dataset = tokenize_data(test_texts, test_labels)

    #
    return train_dataset, val_dataset, test_dataset, AuxLM

    # def compute_metrics
    # computes bleu-4


metric = load_metric("bleu")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def prepare_fine_tuning(auxLM, tokenizer, train_dataset, val_dataset=None, freeze_encoder=False,
                        output_dir='../../../' + paths._get_checkpoint_path()):
    """
    Prepare configurations and base model for fine-tuning
    """
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = auxLM

    if freeze_encoder:
        for param in model.model.encoder.parameters():
            param.requires_grad = False
    print("saving to...", output_dir)
    if val_dataset is not None:
        training_args = TrainingArguments(
            output_dir=output_dir,  # output directory
            num_train_epochs=50,  # total number of training epochs
            per_device_train_batch_size=8,  # batch size per device during training, can increase if memory allows
            per_device_eval_batch_size=1,  # batch size for evaluation, can increase if memory allows
            save_total_limit=1,  # limit the total amount of checkpoints and deletes the older checkpoints
            evaluation_strategy='steps',  # evaluation strategy to adopt during training
            eval_accumulation_steps = 1,
            eval_steps = 500,
            warmup_steps=500,  # number of warmup steps for learning rate scheduler
            weight_decay=0.01,  # strength of weight decay
            logging_dir=output_dir + '/logs',  # directory for storing logs
            logging_steps=100,
            adafactor=True,
            prediction_loss_only = True,
            metric_for_best_model="eval_loss",
            load_best_model_at_end=True,
            group_by_length=True
        )

        trainer = Trainer(
            model=model,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=train_dataset,  # training dataset
            eval_dataset=val_dataset,  # evaluation dataset
            # compute_metrics=compute_metrics,
            callbacks=[transformers.EarlyStoppingCallback(early_stopping_patience=3), ],
            #data_collator=DataCollatorWithPadding(tokenizer, padding=True),
            tokenizer=tokenizer
        )

    else:
        training_args = TrainingArguments(
            output_dir=output_dir,  # output directory
            num_train_epochs=2000,  # total number of training epochs
            per_device_train_batch_size=1,  # batch size per device during training, can increase if memory allows
            save_steps=500,  # number of updates steps before checkpoint saves
            eval_steps=500,
            save_total_limit=5,  # limit the total amount of checkpoints and deletes the older checkpoints
            warmup_steps=500,  # number of warmup steps for learning rate scheduler
            weight_decay=0.01,  # strength of weight decay
            logging_dir='./logs',  # directory for storing logs
            logging_steps=100,
        )

        trainer = Trainer(
            model=model,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=train_dataset,  # training dataset
            tokenizer=tokenizer
        )

    return trainer

if __name__== "__main__":
    # class PegasusPretrain
    logging.info("PREPARING DATA...")
    train_dataset, val_dataset, test_dataset, auxLM = prepare_data()
    logging.info(" FINE-TUNING MODEL...")
    trainer = prepare_fine_tuning(auxLM["model"], auxLM["tokenizer"], train_dataset=train_dataset,
                                  val_dataset=val_dataset)
    trainer.train(resume_from_checkpoint = False)
    trainer.save_model()
    logging.info("EVALUATING...")
    # trainer.evaluate()

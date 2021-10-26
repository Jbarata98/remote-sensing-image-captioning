import json
import re
import statistics

from src.configs.getters.get_data_paths import bleurt_checkpoint
from src.configs.globals import AUX_LM
from src.configs.setters.set_enums import AUX_LMs
from src.configs.setters.set_initializers import CUSTOM_VOCAB
from collections import defaultdict
from bert_score import BERTScorer
from src.metrics_files.bleurt import score as bleurt_sc


def compute_bert_based_scores(test_path, path_results, sentences_generated_path):
    bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)
    bleurt_scorer = bleurt_sc.BleurtScorer(bleurt_checkpoint)

    with open(test_path) as json_file:
        test = json.load(json_file)

    test_sentences = defaultdict(list)
    for ref in test["annotations"]:
        image_id = ref["image_id"]
        caption = ref["caption"]

            #remove leading whitespace
            # print("before", caption)
        test_sentences[image_id].append(caption)

    # get previous score of coco metrics (bleu,meteor,etc) to append bert_based_score
    scores_path = path_results

    with open(scores_path) as json_file:
        scores = json.load(json_file)

    # get previous generated sentences to calculate bertscore according to refs
    generated_sentences_path = sentences_generated_path
    with open(generated_sentences_path) as json_file:
        generated_sentences = json.load(json_file)
    total_precision = 0.0
    total_recall = 0.0
    total_fmeasure = 0.0
    total_bleurt_score = []
    for dict_image_and_caption in generated_sentences:
        image_id = dict_image_and_caption["image_id"]
        #remove leading whitespace
        if AUX_LM == AUX_LMs.GPT2.value and CUSTOM_VOCAB:
            caption = [re.sub(' +', ' ',dict_image_and_caption["caption"].lstrip())]
        else:
            caption = [dict_image_and_caption["caption"]]

        references = [test_sentences[image_id]]
        bleurt_score_per_img = []
        for ref in references[0]:
            bleurt_score_per_img.append(bleurt_scorer.score([ref], caption, batch_size=None)[0])
        total_bleurt_score.append(max(bleurt_score_per_img))

        P_mul, R_mul, F_mul = bert_scorer.score(caption,references)
        precision = P_mul[0].item()
        recall = R_mul[0].item()
        f_measure = F_mul[0].item()

        total_precision += precision
        total_recall += recall
        total_fmeasure += f_measure

        # calculate bert_based_scores
        key_image_id = str(image_id)
        scores[str(key_image_id)]["BertScore_P"] = precision
        scores[key_image_id]["BertScore_R"] = recall
        scores[key_image_id]["BertScore_F"] = f_measure
        scores[key_image_id]["BLEURT"] = max(bleurt_score_per_img)
        # print("\ncaption and score", caption, f_measure)

    n_captions = len(generated_sentences)
    scores["avg_metrics"]["BertScore_P"] = total_precision / n_captions
    scores["avg_metrics"]["BertScore_R"] = total_recall / n_captions
    scores["avg_metrics"]["BertScore_F"] = total_fmeasure / n_captions
    scores["avg_metrics"]["BLEURT"] = statistics.mean(total_bleurt_score)

    print("BERTScore_F:",  total_fmeasure / n_captions)

    print("BLEURT:", statistics.mean(total_bleurt_score))

    # save scores dict to a json
    with open(scores_path, 'w+') as f:
        json.dump(scores, f, indent=2)

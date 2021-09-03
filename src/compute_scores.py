import json
import os

from pycocotools.coco import COCO
import re
from src.configs.setters.set_initializers import *
from src.metrics_files.pycocoevalcap.eval import COCOEvalCap
from src.bert_based_scores import compute_bert_based_scores
from src.configs.setters.set_initializers import *
from datetime import datetime
# EVALUATE = False if the files are generated already, else True
EVALUATE = False

# saving parameters
if os.path.exists('../' + Setters()._set_paths()._get_test_sentences_path()):
    # print("test files stored in:", Setters()._set_paths()._get_test_sentences_path())
    test_files = '../' + Setters()._set_paths()._get_test_sentences_path()

# print("hypothesis files stored in:", '../' + Setters()._set_paths()._get_hypothesis_path())
generated_files = '../' + Setters()._set_paths()._get_hypothesis_path()

print(generated_files)

def create_json(hyp):
    hyp_dict = []
    imgs_index = []
    hyp_list = []
    for i in range(0, len(hyp), 5):  # remove repeated (for each image we had x5) #ensure len is 1094
        hyp_list.append(hyp[i])

    with open(test_files, 'r') as file:
        gts = json.load(file)
    for ref_caps in gts["annotations"]:
        imgs_index.append(ref_caps["image_id"])
    for img, hyp in zip(list(dict.fromkeys(imgs_index)), hyp_list):
        # if using GPT2 needs to preprocess the captions first (removing leading whitespace)
        if CUSTOM_VOCAB and AUX_LM == AUX_LMs.GPT2.value:
            # hack because of whitespaces before word
            x = hyp.lstrip().split(' ')
            hyp_new = ''
            for i in range(0, len(x)):
                if len(x[i - 1]) > 0:
                    hyp_new += x[i]
                else:
                    hyp_new += ' ' + x[i]

            hyp_dict.append({"image_id": img, "caption": hyp_new}) #remove initial space
        else:
            if AUX_LM == AUX_LMs.PEGASUS.value:
                x = hyp.split(' ')
                hyp_new = ''
                for word in x:
                    if len(word) > 0 and word != '.':
                        hyp_new += ' ' + word
                hyp_dict.append({"image_id": img, "caption": hyp_new.lstrip()})
            else:
                hyp_dict.append({"image_id": img, "caption": hyp})
    with open(generated_files, 'w') as fp:
        print('generated file')
        json.dump(hyp_dict, fp)
    return hyp_dict

def compute_scores():
    """
    function to compute the scores
    """

    # declare dict to initialize
    predicted = {}

    # save training details for this experiment
    predicted["training_details"] = Setters()._set_training_parameters()
    # if its pegasus add the multi_input variable


    coco = COCO(test_files)
    cocoRes = coco.loadRes(generated_files)
    #
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params["image_id"] = cocoRes.getImgIds()
    cocoEval.evaluate()

    # save each image score and the avg score to a dict

    individual_scores = [eva for eva in cocoEval.evalImgs]

    # add date
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    predicted["--DATE of COLLECTION--"] = dt_string

    for i in range(len(individual_scores)):
        predicted[individual_scores[i]["image_id"]] = individual_scores[i]
    predicted["avg_metrics"] = cocoEval.eval

    # save scores_dict to a json

    print("storing results files in:", Setters()._set_paths()._get_results_path(bleu_4=predicted["avg_metrics"]["Bleu_4"]))
    output_path =  '../' + Setters()._set_paths()._get_results_path(bleu_4=predicted["avg_metrics"]["Bleu_4"])

    scores_path = output_path
    with open(scores_path, 'w+') as f:
        json.dump(predicted, f, indent=2)

    compute_bert_based_scores(test_path=test_files,
                              path_results=output_path,
                              sentences_generated_path=generated_files)



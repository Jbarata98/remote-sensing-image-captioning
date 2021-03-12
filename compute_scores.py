from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from bert_based_scores import compute_bert_based_scores
from eval import *
import json
from configs.utils import *
EVALUATE = True
# saving parameters

def create_json(hyp):
    hyp_dict= []
    imgs_index = []
    hyp_list = []
    for i in range(0,len(hyp),5):  #remove repeated (for each image we had x5) #ensure len is 1094
         hyp_list.append(hyp[i])

    with open(JSON_test_sentences, 'r') as file:
        gts = json.load(file)
    for ref_caps in gts["annotations"]:
        imgs_index.append(ref_caps["image_id"])
    for img, hyp in zip(list(dict.fromkeys(imgs_index)), hyp_list):
        hyp_dict.append({"image_id": img, "caption": hyp})

    with open(JSON_generated_sentences, 'w') as fp:
        json.dump(hyp_dict, fp)
    return hyp_dict



def main():
    coco = COCO(JSON_test_sentences)
    cocoRes = coco.loadRes(JSON_generated_sentences)
    #
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params["image_id"] = cocoRes.getImgIds()
    cocoEval.evaluate()

    # save each image score and the avg score to a dict
    predicted = {}
    individual_scores = [eva for eva in cocoEval.evalImgs]
    for i in range(len(individual_scores)):
        predicted[individual_scores[i]["image_id"]] = individual_scores[i]
    predicted["avg_metrics"] = cocoEval.eval

    # save scores dict to a json
    scores_path = evaluation_results
    with open(scores_path, 'w+') as f:
        json.dump(predicted, f, indent=2)

    compute_bert_based_scores(test_path = JSON_test_sentences,
                      path_results = evaluation_results,
                      sentences_generated_path= JSON_generated_sentences)


# if EVALUATE:
#     # refs, hyps = evaluate(beam_size)
#     # create_json(hyps)

main()


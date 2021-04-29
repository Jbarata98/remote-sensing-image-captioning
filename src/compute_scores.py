from pycocotools.coco import COCO
from src.configs.initializers import *
from src.metrics_files.pycocoevalcap.eval import COCOEvalCap
from src.bert_based_scores import compute_bert_based_scores
from src.configs.initializers import PATHS, h_parameter
from eval import evaluator

# EVALUATE = False if the files are generated already, else True
EVALUATE = False

# saving parameters
if os.path.exists(PATHS._get_test_sentences_path()):
    print("test files stored in:", PATHS._get_test_sentences_path())
    test_files = PATHS._get_test_sentences_path()

print("hypothesis files stored in:", PATHS._get_hypothesis_path())
generated_files = PATHS._get_hypothesis_path()


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
        hyp_dict.append({"image_id": img, "caption": hyp})

    with open(generated_files, 'w') as fp:
        json.dump(hyp_dict, fp)
    return hyp_dict


def main():
    # declare dict to initialize
    predicted = {}

    # save training details for this experiment
    predicted["training_details"] = h_parameter

    coco = COCO(test_files)
    cocoRes = coco.loadRes(generated_files)
    #
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params["image_id"] = cocoRes.getImgIds()
    cocoEval.evaluate()

    # save each image score and the avg score to a dict

    individual_scores = [eva for eva in cocoEval.evalImgs]
    for i in range(len(individual_scores)):
        predicted[individual_scores[i]["image_id"]] = individual_scores[i]
    predicted["avg_metrics"] = cocoEval.eval

    # save scores_dict to a json

    print("storing results files in:", PATHS._get_results_path(bleu_4=predicted["avg_metrics"]["Bleu_4"]))
    output_path = PATHS._get_results_path(bleu_4=predicted["avg_metrics"]["Bleu_4"])

    scores_path = output_path
    with open(scores_path, 'w+') as f:
        json.dump(predicted, f, indent=2)

    compute_bert_based_scores(test_path=test_files,
                              path_results=output_path,
                              sentences_generated_path=generated_files)


# if want to generate hypotheses and references array
if EVALUATE:

    eval = evaluator(device=DEVICE)

    # load checkpoint
    eval._load_checkpoint()

    # evaluate the current checkpoint model
    refs, hyps = eval._evaluate()

    # create json files
    create_json(hyps)

# if already evaluated
else:
    # load hypothesis path
    with open(PATHS._get_hypothesis_path(results_array=True), "rb") as hyp_file:
        hyps = pickle.load(hyp_file)

    create_json(hyps)

main()

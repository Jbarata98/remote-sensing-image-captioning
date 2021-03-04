from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice
from bert_score import BERTScorer

from eval import *
from bleurt import score as bleurt_sc
import statistics


EVALUATE = False
ATTENTION = 'soft_attention' #TODO hard_attention
JSON_results = 'hypothesis'
JSON_refs = 'references'
bleurt_checkpoint = "bleurt/test_checkpoint" #uses Tiny Bleurt
out_file = open("evaluation_results_" + ATTENTION + ".txt" , "w")

def create_json(hyp,refs):
    hyp_dict, ref_dict = {},{}
    img_count_refs,img_count_hyps = 0,0
    for ref_caps in refs:
        ref_dict[str(img_count_refs)] = [item for sublist in ref_caps for item in sublist]
        img_count_refs+=1
    for hyp_caps in hyp:
        hyp_dict[str(img_count_hyps)] = [hyp_caps]
        img_count_hyps+=1

    with open(JSON_results + '.json', 'w') as fp:
        json.dump(hyp_dict, fp)
    with open(JSON_refs +'.json', 'w') as fp:
        json.dump(ref_dict, fp)

def open_json():
    with open(JSON_refs + '.json', 'r') as file:
        gts = json.load(file)
    with open(JSON_results + '.json', 'r') as file:
        res = json.load(file)

    return gts,res

def bleu(gts,res):
    scorer = Bleu(n=4)

    score, scores = scorer.compute_score(gts, res)

    out_file.write('BLEU(1-4) = %s' % score + '\n')


def cider(gts,res):
    scorer = Cider()
    (score, scores) = scorer.compute_score(gts, res)
    out_file.write('CIDEr = %s' % score + '\n')

def meteor(gts,res):
    scorer = Meteor()
    score, scores = scorer.compute_score(gts, res)
    out_file.write('METEOR = %s' % score + '\n')

def rouge(gts,res):
    scorer = Rouge()
    score, scores = scorer.compute_score(gts, res)
    out_file.write('ROUGE = %s' % score + '\n')

def spice(gts, res):
    scorer = Spice()
    score, scores = scorer.compute_score(gts, res)
    out_file.write('SPICE = %s' % score + '\n')

def bert_based(gts,res):
    refs, cands = [], []
    for refers in gts.values():
        sub_refs = []
        for ref in refers:
            sub_refs.append(ref + '.')
        refs.append(sub_refs)
    for cand in res.values():
        cands.append(cand[0] + '.')

    scorer = BERTScorer(lang="en", rescale_with_baseline=True)
    P, R, F1 = scorer.score(cands, refs, verbose=True)
    out_file.write('BERTScore = %s' % F1.mean().item() + "\n")
    BERTScore = F1.mean().item()

    total_bleurt_score = []
    scorer = bleurt_sc.BleurtScorer(bleurt_checkpoint)

    for ref_caption,cand in zip(refs,cands):
        bleurt_score_per_img = []
        for ref in ref_caption:
            bleurt_score_per_img.append(scorer.score([ref], [cand], batch_size=None)[0])
        total_bleurt_score.append(max(bleurt_score_per_img))
    out_file.write('BLEURT =%s' % statistics.mean(total_bleurt_score))

def main():
    gts,res = open_json()
    bleu(gts,res)
    cider(gts,res)
    meteor(gts,res)
    rouge(gts,res)
    spice(gts,res)
    bert_based(gts,res)
    out_file.close()

if EVALUATE:

    refs, hyps = evaluate(beam_size)
    create_json(hyps, refs)

main()












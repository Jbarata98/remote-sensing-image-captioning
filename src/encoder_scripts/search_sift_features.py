import json
import pickle
import faiss
import heapq
#todo REFACTOR **NOT WORKING PROPERLY**
import numpy as np
from PIL import Image
from numpy import long
from tqdm import tqdm
from sift_feature_extractor import calc_sift, get_sift

features_list = feature_list = pickle.load(open('/home/starksultana/Documentos/MEIC/5o_ano/Tese/code/remote-sensing-image-captioning/experiments/encoder/features/SIFT_rsicd_features.pickle', 'rb'))

with open('/home/starksultana/Documentos/MEIC/5o_ano/Tese/code/remote-sensing-image-captioning/experiments/encoder/indexes/sift_dict_rsicd', "rb") as dict_file:
    index_dict = pickle.load(dict_file)

index = faiss.read_index('/home/starksultana/Documentos/MEIC/5o_ano/Tese/code/remote-sensing-image-captioning/experiments/encoder/indexes/sift_index_rsicd')
sift = get_sift()
def neighbor_dict(id_, score):
    return {'id': long(id_), 'score': score}


def result_dict_str(id_, neighbors):
    return {'id': id_, 'neighbors': neighbors}

results = []

def _search_(ids, vectors, topN):
    # for id,(filename,sift_feature) in tqdm(index_dict.items()):
    for id,sift_feature in zip(ids,tqdm(vectors)):

        scores, neighbors = index.search(sift_feature, k = topN) if sift_feature.size > 0 else ([], [])
        n, d = neighbors.shape
        result_dict = {}

        for i in range(n):
            l = np.unique(neighbors[i]).tolist()
            for r_id in l:
                if r_id != -1:
                    score = result_dict.get(r_id, 0)
                    score += 1
                    result_dict[r_id] = score

        result_dict = {k: v for k, v in sorted(result_dict.items(), key=lambda item: item[1])}


        neighbors_scores = []
        for e in result_list:
            confidence = e[0] * 100 / n
            # if id_to_vector:
            #     file_path = id_to_vector(e[1])[0]
            #     neighbors_scores.append(neighbor_dict_with_path(e[1], file_path, str(confidence)))
            neighbors_scores.append(neighbor_dict(e[1], str(confidence)))
        results.append(result_dict_str(id, neighbors_scores))
    return results
# with open('results_sift.json', 'w') as outfile:
#     json.dump(results, outfile)
#
# with open('results_sift.json') as json_file:
#     data = json.load(json_file)
#     #
# print(data)
def get_vectors(sift, image):
    # if is base64 string
    # image = save_tmp_image(image)
    # if is image path
    return calc_sift(sift, image)

def search_by_image(image, k):
    ids = [None]
    ret, vectors = get_vectors(sift, image)
    results = _search_(ids, [vectors], topN=k)
    return results

results = search_by_image('/home/starksultana/Documentos/MEIC/5o_ano/Tese/code/remote-sensing-image-captioning/data/images/flickr8k/57417274_d55d34e93e.jpg',k =5)
print(results)
# print(index_dict.get(3705))
img = Image.open('/home/starksultana/Documentos/MEIC/5o_ano/Tese/code/remote-sensing-image-captioning/data/images/flickr8k/57417274_d55d34e93e.jpg')
img.show()
img = Image.open(index_dict.get(7599)[0])
img.show()
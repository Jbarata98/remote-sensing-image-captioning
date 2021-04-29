import numpy as np
import os
import cv2
import logging
import uuid
import sys
import pickle
from tqdm import tqdm
import faiss
#create a opencv sift extractor
def get_sift():
    return cv2.xfeatures2d.SIFT_create()
#calculate sift
def calc_sift(sift, image_file):
    if not os.path.isfile(image_file):
        logging.error('Image:{} does not exist'.format(image_file))
        return -1, None

    try:
        image_o = cv2.imread(image_file)
    except:
        logging.error('Open Image:{} failed'.format(image_file))
        return -1, None

    if image_o is None:
        logging.error('Open Image:{} failed'.format(image_file))
        return -1, None

    image = cv2.resize(image_o, (224, 224))
    if image.ndim == 2:
        gray_image = image
    else:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    kp, des = sift.detectAndCompute(gray_image, None)
    # print("calculating sift...")
    sift_feature = np.matrix(des)
    return 0, sift_feature



if __name__ == '__main__':
    INDEX_KEY = "IDMap,IMI2x10,Flat"
    index = faiss.index_factory(128, INDEX_KEY)
    root_path = "/home/starksultana/Documentos/MEIC/5o_ano/Tese/code/remote-sensing-image-captioning/data/images/flickr8k"
    # features = []
    index_dict = {}
    ids = None
    features = np.matrix([])


    for root, dirs, files in os.walk(root_path):
        with tqdm(total=len(files)) as pbar:
            for id, filename in enumerate(files):
                file_name = root_path+ '/' + filename
                sift = get_sift()
                ret, sift_feature = calc_sift(sift,file_name)
                if ret == 0 and sift_feature.any():
                    # record id and path
                    image_dict = {id: (file_name, sift_feature)}
                    index_dict.update(image_dict)
                    # print ids_count
                    # print sift_feature.shape[0]
                    ids_list = np.linspace(id, id, num=sift_feature.shape[0], dtype="int64")
                    if features.any():
                        features = np.vstack((features, sift_feature))
                        ids = np.hstack((ids, ids_list))
                    else:
                        features = sift_feature
                        ids = ids_list
                    if id % 500 == 499:
                        if not index.is_trained and INDEX_KEY != "IDMap,Flat":
                            index.train(features)
                        index.add_with_ids(features, ids)
                        ids = None
                        features = np.matrix([])
                    pbar.update(1)

    if features.any():
        if not index.is_trained and INDEX_KEY != "IDMap,Flat":
            index.train(features)
        index.add_with_ids(features, ids)

        # save index
    faiss.write_index(index, '/home/starksultana/Documentos/MEIC/5o_ano/Tese/code/remote-sensing-image-captioning/experiments/encoder/indexes/sift_index_rsicd')

    # save ids
    with open('/home/starksultana/Documentos/MEIC/5o_ano/Tese/code/remote-sensing-image-captioning/experiments/encoder/indexes/sift_dict_rsicd', 'wb+') as f:
        try:
            pickle.dump(index_dict, f, True)
        except EnvironmentError as e:
            logging.error('Failed to save index file error:[{}]'.format(e))
            f.close()
    f.close()
                    # features.append(sift_feature)









    # # pickle.dump(features, open('/home/starksultana/Documentos/MEIC/5o_ano/Tese/code/remote-sensing-image-captioning/experiments/encoder/features/SIFT_rsicd_features.pickle', 'wb'))
    # #
    # features_list = feature_list = pickle.load(open('/home/starksultana/Documentos/MEIC/5o_ano/Tese/code/remote-sensing-image-captioning/experiments/encoder/features/SIFT_rsicd_features.pickle', 'rb'))


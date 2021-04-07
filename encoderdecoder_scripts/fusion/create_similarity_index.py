from configs.get_data_paths import *
from configs.globals import *
import faiss


PATHS = Paths()

features_list = feature_list = pickle.load(open(PATHS._get_features_path('TRAIN'), 'rb'))


def flatten_maps(feature_list):
    f_maps = []
    # flatten the feature maps
    for fmap in feature_list:
        fmap = fmap.flatten(start_dim=0, end_dim=2)  # (1,7,7,2048) feature map

        f_maps.append(np.ascontiguousarray(fmap))  # (49,2048) dimension feature map
    return f_maps


def get_image_name():
    # get captions path to retrieve image name

    file = open('../' + PATHS._get_captions_path())
    data = json.load(file)
    train_filenames = []
    for image in data['images']:
        if image['split'] == "train":
            train_filenames.append(image['filename'])

    return train_filenames


def create_index(feature_list):
    feature_maps = flatten_maps(feature_list)

    dimensions = feature_maps[0].shape[1]  # 2048

    image_files = get_image_name()

    index = faiss.index_factory(dimensions, "IDMap,Flat")

    index_dict = {}

    for id, (feature, image_name) in enumerate(zip(tqdm(feature_maps), image_files)):

        image_dict = {id: image_name}

        index_dict.update(image_dict)

        ids_list = np.linspace(id, id, num=feature.shape[0], dtype="int64")

        # index.train(feature)

        index.add_with_ids(feature, ids_list)

    return index, index_dict


if __name__ == '__main__':
    index, index_dict = create_index(feature_list)
    print("Writing index...")
    faiss.write_index(index, PATHS._get_index_path()['path_index'])
    print("Writing dict...")
    with open(PATHS._get_index_path()['path_dict'], 'wb+') as f:
        pickle.dump(index_dict, f, True)

    f.close()

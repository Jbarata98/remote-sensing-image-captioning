import faiss
import torch
from torch import nn
from tqdm import tqdm

from src.configs.getters.get_data_paths import Paths
from src.configs.globals import *

batch_size = 32
workers = 1

class ImageRetrieval():

    def __init__(self, dim_examples, encoder, train_dataloader_images, device):
        #print("self dim exam", dim_examples)
        self.datastore = faiss.IndexFlatL2(dim_examples) #datastore
        self.encoder= encoder

        #data
        self.device=device
        self.imgs_indexes_of_dataloader = torch.tensor([]).long().to(device)
        #print("self.imgs_indexes_of_dataloader type", self.imgs_indexes_of_dataloader)

        #print("len img dataloader", self.imgs_indexes_of_dataloader.size())
        # self._add_examples(train_dataloader_images)
        #print("len img dataloader final", self.imgs_indexes_of_dataloader.size())
        #print("como ficou img dataloader final", self.imgs_indexes_of_dataloader)


    def _add_examples(self, train_dataloader_images):
        print("\nadding input examples to datastore (retrieval)")
        self.image_model, self.dim = self.encoder._get_encoder_model()
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

        for p in self.image_model.parameters():
            p.requires_grad = False

        for i, (imgs, imgs_indexes) in enumerate(tqdm(train_dataloader_images)):
            #add to the datastore
            imgs=imgs.to(self.device)
            imgs_indexes = imgs_indexes.long().to(self.device)
            #print("img index type", imgs_indexes)
            encoder_output = self.image_model.Extract_features(imgs)

            out = self.adaptive_pool(encoder_output)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
            out = out.permute(0, 2, 3, 1)  # (ba

            encoder_output = out.view(out.size()[0], -1, out.size()[-1])
            # print(encoder_output.shape)
            input_img = encoder_output.mean(dim=1)

            # for img in input_img:
            # print(input_img.shape)
            self.datastore.add(input_img.cpu().numpy())

            if i%5==0:
                print("i and img index of ImageRetrival",i, imgs_indexes)
                print("n of examples", self.datastore.ntotal)
            self.imgs_indexes_of_dataloader= torch.cat((self.imgs_indexes_of_dataloader,imgs_indexes))

        faiss.write_index(self.datastore,'index_rita')

    def retrieve_nearest_for_train_query(self, query_img, k=2):
        self.datastore = faiss.read_index('index_rita')
        #print("self query img", query_img)
        D, I = self.datastore.search(query_img, k)     # actual search
        # print(I[0][1])
        #print("all nearest", I)
        #print("I firt", I[:,0])
        #print("if you choose the first", self.imgs_indexes_of_dataloader[I[:,0]])
        # print(I)
        for neighbors in I:
            nearest_input = (neighbors[0],neighbors[1])
        #print("the nearest input is actual the second for training", nearest_input)
        #nearest_input = I[0,1]
        #print("actual nearest_input", nearest_input)
        return nearest_input

    def retrieve_nearest_for_val_or_test_query(self, query_img, k=1):
        D, I = self.datastore.search(query_img, k)     # actual search
        nearest_input = self.imgs_indexes_of_dataloader[I[:,0]]
        #print("all nearest", I)
        #print("the nearest input", nearest_input)
        return nearest_input

PATHS = Paths(encoder=ENCODER_MODEL)
print(ENCODER_MODEL)
#
# ENCODER = Encoders(model=ENCODER_MODEL,
#                    checkpoint_path='../' + PATHS._load_encoder_path(encoder_loader=ENCODER_LOADER, augment=True), device=DEVICE)
# train_retrieval_loader = torch.utils.data.DataLoader(
#     TrainRetrievalDataset(data_folder = '/home/starksultana/Documentos/MEIC/5o_ano/Tese/code/remote-sensing-image-captioning/data/paths'),
#     batch_size=batch_size, shuffle=False, num_workers=workers)#, pin_memory=True)
# image_retrieval = ImageRetrieval(2048, ENCODER, train_retrieval_loader, DEVICE)
#
# train_retrieval_loader_test = torch.utils.data.DataLoader(
#     TrainRetrievalDataset(data_folder = '/home/starksultana/Documentos/MEIC/5o_ano/Tese/code/remote-sensing-image-captioning/data/paths'),
#     batch_size=32, shuffle=False, num_workers=workers)#, pin_memory=True)
#
# neighbors = []
# with torch.no_grad():
#     image_model, dim = ENCODER._get_encoder_model()
#     adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
#     for i,img in enumerate(tqdm(train_retrieval_loader_test)):
#         imgs = img[0].to(DEVICE)
#
#         # Forward prop.
#         encoder_output = image_model.extract_features(imgs)
#         out = adaptive_pool(encoder_output)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
#         out = out.permute(0, 2, 3, 1)  # (ba
#
#         imgs = out.view(out.size()[0], -1, out.size()[-1])
#         #print("this was the imgs out", imgs.size())
#         input_imgs = imgs.mean(dim=1)
#
#         nearest_imgs = image_retrieval.retrieve_nearest_for_train_query(input_imgs.cpu().numpy())
#         neighbors.append(nearest_imgs)
#
#     pickle.dump(neighbors, open('features-rita.pickle', 'wb'))




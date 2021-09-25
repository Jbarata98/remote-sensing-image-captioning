from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms
from tqdm import tqdm

from src.classification_scripts.set_classification_globals import _set_globals
from src.classification_scripts.SupConLoss.SupConModel import LinearClassifier

from src.configs.setters.set_initializers import *

import faiss
import pickle
import numpy as np

from src.image_retrieval.cnn_feature_extractor import ENCODER

classifier = LinearClassifier(eff_net_version='v2')
checkpoint = torch.load('../../experiments/encoder/encoder_checkpoints/SupConClassifier.pth.tar')
classifier.load_state_dict(checkpoint['classifier'])


tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
img = tfms(Image.open('../../data/images/RSICD_images/center_154.jpg')).unsqueeze(0)

image_model, dim = ENCODER._get_encoder_model(eff_net_version='v2')

feat = image_model.forward_features(img)


 # torch.Size([1, 3, 224, 224])


classifier.eval()
with torch.no_grad():
    # print(feat.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2).mean(dim=1).detach().shape)
    vec = classifier(feat.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2).mean(dim=1).detach())
    # print(vec)
    y = torch.argmax(vec, dim=1)
    label = y.item()
    print(label)
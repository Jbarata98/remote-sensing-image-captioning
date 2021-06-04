import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet


class SupConEffNet(nn.Module):
    """backbone + projection head"""

    def __init__(self, head='mlp', feat_dim=128):
        super(SupConEffNet, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b5')
        self.encoder_dim = self.model._fc.in_features
        if head == 'linear':
            self.head = nn.Linear(self.encoder_dim, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(self.encoder_dim, self.encoder_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.encoder_dim, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.model.extract_features(x)
        feat = feat.permute(0, 2, 3, 1).flatten(start_dim = 1, end_dim=2).mean(dim=1)

        feat = F.normalize(self.head(feat), dim=1)
        return feat


class LinearClassifier(nn.Module):
    """Linear classifier"""

    def __init__(self, num_classes=31):
        super(LinearClassifier, self).__init__()
        image_model = EfficientNet.from_pretrained('efficientnet-b5')
        encoder_dim = image_model._fc.in_features
        self.fc = nn.Linear(encoder_dim, num_classes)

    def forward(self, features):

        return self.fc(features)

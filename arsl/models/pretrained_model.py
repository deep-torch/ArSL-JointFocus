import torch.nn as nn
from torchvision import models


class PretrainedModel(nn.Module):

    def __init__(self):
        super(PretrainedModel,self).__init__()

        self.feature_extractor = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT).features
        num_layers_to_freeze = int(len(self.feature_extractor) * 0.9)
        self.feature_extractor[:num_layers_to_freeze].requires_grad_(False)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.lstm = nn.LSTM(1280, 256, batch_first=True)

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512, 502),
        )

    def forward(self, x):
        batch_size, num_frames, c, h, w = x.size()
        x = x.view(batch_size * num_frames, c, h, w)
        
        x = self.feature_extractor(x)
        x = self.adaptive_pool(x)

        x = x.view(batch_size, num_frames, -1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]

        logits = self.linear_relu_stack(x)
        return logits

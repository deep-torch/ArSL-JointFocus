import torch.nn as nn


class BaselineModel(nn.Module):
    def __init__(self):
        super(BaselineModel,self).__init__()

        self.cnn_stack=self.building_CNN(5,16)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.lstm = nn.LSTM(512, 256, batch_first=True)

        self.linear_relu_stack = nn.Sequential(
          nn.Linear(256, 512),
          nn.ReLU(),
          nn.Linear(512,512),
          nn.ReLU(),
          nn.Linear(512, 502),
        )

    def building_CNN(self,num_block,inp_filters):
        model=nn.Sequential()

        model.add_module('initial_conv', nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1))
        model.add_module('initial_batchnorm', nn.BatchNorm2d(16))
        model.add_module('initial_relu', nn.ReLU())

        inp_chanels = 16
        layers = 3

        for i in range(num_block):
          filters = [inp_filters * (2 ** i), inp_filters * (2 ** i), inp_filters * (2 ** (i + 1))]
          stride_parameter = [1, 1, 2]
          padding_parameter = [1, 1, 0]
          for j in range(layers):
            model.add_module(f'conv{i+1}_{j+1}', nn.Conv2d(inp_chanels, filters[j], kernel_size=3, stride=stride_parameter[j], padding=padding_parameter[j]))
            model.add_module(f'batchnorm{i+1}_{j+1}', nn.BatchNorm2d(filters[j]))
            model.add_module(f'relu{i+1}_{j+1}', nn.ReLU())
            inp_chanels=filters[j]

        return model

    def forward(self, x):
        batch_size, num_frames, c, h, w = x.size()
        x = x.view(batch_size * num_frames, c, h, w)

        x = self.cnn_stack(x)
        
        # You can use the line below instead of the previous one to reduce GPU memory usage
        # x = checkpoint_sequential(self.cnn_stack, segments=5, input=x)
        x = self.adaptive_pool(x)

        x = x.view(batch_size, num_frames, -1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]

        logits = self.linear_relu_stack(x)
        return logits

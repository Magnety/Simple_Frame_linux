import torch
import torch.nn as nn

cfg = {
    'A' : [32,     'M1', 64,      'M1', 128, 128,           'M2', 256, 256,           'M2', 256, 256,           'M2'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M1', 128, 128, 'M2', 256, 256, 256,      'M2', 512, 512, 512,      'M2', 512, 512, 512,      'M2'],
    'E' : [64, 64, 'M1', 128, 128, 'M2', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):

    def __init__(self, features, num_class=2):
        super().__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(359, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, num_class),
        )
        self.extra = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)),nn.Dropout(0.5))

    def forward(self, x,feature):
       # print("///////feature//////////")
        #print(feature.shape)
        output = self.features(x)
        extra = self.extra(output)
        extra = extra.view(extra.size()[0], -1)
        #print("///////extra//////////")
        #print(extra.shape)
        f = torch.cat((feature,extra),dim=-1)
        #print("extra.shape:",extra.shape)
        #print("feature.shape:",feature.shape)
        #print("f.shape:",f.shape)
        output = self.classifier(f)
        return output

def make_layers(cfg, batch_norm=False):
    layers = []
    input_channel = 1
    for l in cfg:
        if l == 'M1':
            layers += [nn.MaxPool3d(kernel_size=2, stride=2)]
            continue
        if l == 'M2':
            layers += [nn.MaxPool3d(kernel_size=(1,2,2), stride=2)]
            continue

        layers += [nn.Conv3d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm3d(l)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l

    return nn.Sequential(*layers)

def vgg11_bn():
    return VGG(make_layers(cfg['A'], batch_norm=True))

def vgg13_bn():
    return VGG(make_layers(cfg['B'], batch_norm=True))

def vgg16_bn():
    return VGG(make_layers(cfg['D'], batch_norm=True))

def vgg19_bn():
    return VGG(make_layers(cfg['E'], batch_norm=True))
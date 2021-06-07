from simple_frame.network_architecture.neural_network import ClassficationNetwork
from simple_frame.network_architecture.generic_VNet import VNet_class
from simple_frame.network_architecture.myResNet import resnext50_32x4d
import torch
from torch import Tensor
import torch.nn as nn
class Ensemble(ClassficationNetwork):

    def __init__(
        self,
            deep_supervision=True,
            num_classes = 2,
    ) -> None:
        super(Ensemble, self).__init__()
        self.do_ds = False
        self.conv_op = nn.Conv3d
        self.norm_op = nn.BatchNorm3d
        self.dropout_op = nn.Dropout3d
        self.num_classes = num_classes
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision
        self.vnet = VNet_class(num_classes=self.num_classes, deep_supervision=True)
        self.resnet = resnext50_32x4d(num_classes=self.num_classes)


    def forward(self, x):
        out1 = self.vnet(x)
        out2 = self.resnet(x)
        out = (out1+out2)/2
        return out
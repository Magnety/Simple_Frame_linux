import copy
import random
from functools import wraps
from batchgenerators.transforms import DataChannelSelectionTransform, SegChannelSelectionTransform, SpatialTransform, \
    GammaTransform, MirrorTransform, Compose
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, \
    ContrastAugmentationTransform, BrightnessTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor
from simple_frame.data_augmentation.default_data_augmentation import default_3D_augmentation_params
from simple_frame.network_architecture.neural_network import ClassficationNetwork
from simple_frame.byol.aug3d import RandCrop_tu,Gaussiannoise_tu,Mirror_tu,Spatial_tansform_tu
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms as T

# helper functions
class ResNeXt3D(nn.Module):

    def __init__(self, block, layers, shortcut_type='B', cardinality=32, num_classes=400):
        self.inplanes = 64
        super(ResNeXt3D, self).__init__()
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        # self.bn1 = nn.BatchNorm3d(64)
        self.gn1 = nn.GroupNorm(32, 64)
        self.relu = nn.PReLU()
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type, cardinality)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, cardinality, stride=(1, 2, 2))
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, cardinality, stride=1)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, cardinality, stride=1)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(1024, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def _make_layer(self, block, planes, blocks, shortcut_type, cardinality, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                # nn.BatchNorm3d(planes * block.expansion)
                nn.GroupNorm(32, planes * block.expansion),
            )
        layers = []
        layers.append(
            block(self.inplanes, planes, cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, cardinality))

        return nn.Sequential(*layers)

    def forward(self, x):
        #print("RES:x.shape:",x.shape)
        x = self.conv1(x)
        #print("RES:x.shape:",x.shape)
        x = self.gn1(x)
        #print("RES:x.shape:",x.shape)
        x = self.relu(x)
        #print("RES:x.shape:",x.shape)
        x = self.maxpool(x)
        #print("RES:x.shape:",x.shape)


        x = self.layer1(x)
        #print("RES:x.shape:",x.shape)

        x = self.layer2(x)
        #print("RES:x.shape:",x.shape)

        x = self.layer3(x)
        #print("RES:x.shape:",x.shape)
        x = self.layer4(x)
        #print("RES:x.shape:",x.shape)
        x = self.avgpool(x)
        #print("RES:x.shape:",x.shape)
        x = x.view(x.size(0), -1)
        #print("RES:x.shape:",x.shape)
        ###print("RES:x.shape:",x.shape)
        return x
class ResNeXtBottleneck(nn.Module):
    expansion = 2
    def __init__(self, inplanes, planes, cardinality, stride=1,
                 downsample=None):
        super(ResNeXtBottleneck, self).__init__()
        mid_planes = cardinality * int(planes / 32)
        self.conv1 = nn.Conv3d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.gn1 = nn.GroupNorm(32, mid_planes)
        # self.bn1 = nn.BatchNorm3d(mid_planes)
        self.conv2 = nn.Conv3d(
            mid_planes,
            mid_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False)
        self.gn2 = nn.GroupNorm(32, mid_planes)
        # self.bn2 = nn.BatchNorm3d(mid_planes)
        self.conv3 = nn.Conv3d(
            mid_planes, planes * self.expansion, kernel_size=1, bias=False)
        # self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.gn3 = nn.GroupNorm(32, planes * self.expansion)
        self.relu = nn.PReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.gn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.gn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
def default(val, def_val):
    return def_val if val is None else val

def flatten(t):
    return t.reshape(t.shape[0], -1)

def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn

def get_module_device(module):
    return next(module.parameters()).device

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

# loss fn

def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

# augmentation utils

class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)

# exponential moving average

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

# MLP class for projector and predictor

class MLP(nn.Module):
    def __init__(self, dim, projection_size, hidden_size = 4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)

# a wrapper class for the base neural network
# will manage the interception of the hidden layer output
# and pipe it into the projecter and predictor nets

class NetWrapper(nn.Module):
    def __init__(self, net, projection_size, projection_hidden_size, layer = -2):
        super().__init__()
        self.net = net
        self.layer = layer   # -2
        self.projector = None
        self.projection_size = projection_size  #256
        self.projection_hidden_size = projection_hidden_size   #4096
        self.hidden = {}
        self.hook_registered = False

    def _find_layer(self):
        if type(self.layer) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*self.net.children()]
            return children[self.layer]
        return None

    def _hook(self, _, input, output):
        device = input[0].device
        self.hidden[device] = flatten(output)

    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f'hidden layer ({self.layer}) not found'
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    @singleton('projector')
    def _get_projector(self, hidden):
        _, dim = hidden.shape
        projector = MLP(dim, self.projection_size, self.projection_hidden_size)
        return projector.to(hidden)

    def get_representation(self, x):
        if self.layer == -1:
            return self.net(x)

        if not self.hook_registered:
            self._register_hook()

        self.hidden.clear()
        _ = self.net(x)
        hidden = self.hidden[x.device]
        self.hidden.clear()

        assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
        return hidden

    def forward(self, x, return_projection = True):
        print("NW:x.shape:",x.shape)
        representation = self.get_representation(x)
        print("NW:representation.shape:", representation.shape)
        if not return_projection:
            return representation

        projector = self._get_projector(representation)
        projection = projector(representation)
        return projection, representation

# main class

class BYOL(SegmentationNetwork):
    def __init__(
        self,
        num_classes,deep_supervision,image_size=(24,128,128),
        hidden_layer = -1,
        projection_size = 256,
        projection_hidden_size = 4096,
        augment_fn = None,
        augment_fn2 = None,
        moving_average_decay = 0.99,
        use_momentum = True,


    ):
        super().__init__()
        self.net =  ResNeXt3D(ResNeXtBottleneck, [3, 4, 6, 3], num_classes=2)
        self.do_ds = False
        norm_cfg = 'BN'
        activation_cfg = 'ReLU'
        self.conv_op = nn.Conv3d
        self.norm_op = nn.BatchNorm3d
        self.dropout_op = nn.Dropout3d
        self.num_classes = num_classes
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision
        # default SimCLR augmentation


        DEFAULT_AUG = nn.Sequential(
            RandCrop_tu(image_size=image_size,crop_size=(8,64,64)),
            RandomApply(
            Gaussiannoise_tu(image_size=(8,64,64),SNR=20),
            p=0.5
            ),
            RandomApply(
                Mirror_tu(image_size=(8, 64, 64)),
                p=0.5
            ),
            RandomApply(
                Spatial_tansform_tu(image_size=(8, 64, 64)),
                p=0.5
            ),
        )

        self.augment1 = default(augment_fn, DEFAULT_AUG)
        self.augment2 = default(augment_fn2, self.augment1)

        self.online_encoder = NetWrapper(self.net, projection_size, projection_hidden_size, layer=hidden_layer)

        self.use_momentum = use_momentum
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)

        self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size)

        # get device of network and make wrapper same device
        device = get_module_device(self.net)
        self.to(device)

        # send a mock image tensor to instantiate singleton parameters
        #self.forward(torch.randn(2, 3, image_size[0], image_size[1],image_size[2], device=device))

    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(target_encoder, False)
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    def forward(self,x,return_embedding = False,return_projection = True):
        if return_embedding:
            return self.online_encoder(x, return_projection = return_projection)
        #print("//////////////////////////////")
        #print("x.shape",x.shape)
        image_one, image_two = self.augment1(x),self.augment2(x)
        #print("image_one.shape",image_one.shape)
        online_proj_one, _ = self.online_encoder(image_one)
        online_proj_two, _ = self.online_encoder(image_two)

        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)
        with torch.no_grad():
            target_encoder = self._get_target_encoder() if self.use_momentum else self.online_encoder
            target_proj_one, _ = target_encoder(image_one)
            target_proj_two, _ = target_encoder(image_two)
            target_proj_one.detach_()
            target_proj_two.detach_()

        loss_one = loss_fn(online_pred_one, target_proj_two.detach())
        loss_two = loss_fn(online_pred_two, target_proj_one.detach())

        loss = loss_one + loss_two
        return loss.mean()
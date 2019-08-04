import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models.resnet import model_zoo, model_urls
import attr


@attr.s
class ResnetNoStageConfig(object):
    image_channels = 3
    num_layers = 34
    num_deconv_layers = 3
    num_deconv_filters = 256
    num_deconv_kernel = 4
    final_conv_kernel = 1
    # Need to modifiy the num_keypoints to match the required value
    num_keypoints = -1

    # For 2d keypoint, this value should be one
    # For 2d keypoint + depth, this value should be 2
    # For 3d volumetric keypoint, this value should be the depth resolution
    depth_per_keypoint = 1


# The specification of resnet
resnet_spec = {18: (BasicBlock, [2, 2, 2, 2], [64, 64, 128, 256, 512], 'resnet18'),
               34: (BasicBlock, [3, 4, 6, 3], [64, 64, 128, 256, 512], 'resnet34'),
               50: (Bottleneck, [3, 4, 6, 3], [64, 256, 512, 1024, 2048], 'resnet50'),
               101: (Bottleneck, [3, 4, 23, 3], [64, 256, 512, 1024, 2048], 'resnet101')}


class DeconvHead(nn.Module):
    def __init__(self,
                 in_channels,  # type: int
                 num_layers,  # type: int
                 num_filters,  # type: int
                 kernel_size,  # type: int
                 conv_kernel_size,  # type: int
                 num_joints,  # type: int
                 depth_dim,  # type: int
                 with_bias_end=True):
        super(DeconvHead, self).__init__()

        conv_num_filters = num_joints * depth_dim

        assert kernel_size == 2 or kernel_size == 3 or kernel_size == 4, 'Only support kenerl 2, 3 and 4'
        padding = 1
        output_padding = 0
        if kernel_size == 3:
            output_padding = 1
        elif kernel_size == 2:
            padding = 0

        assert conv_kernel_size == 1 or conv_kernel_size == 3, 'Only support kenerl 1 and 3'
        if conv_kernel_size == 1:
            pad = 0
        elif conv_kernel_size == 3:
            pad = 1

        self.features = nn.ModuleList()
        for i in range(num_layers):
            _in_channels = in_channels if i == 0 else num_filters
            self.features.append(
                nn.ConvTranspose2d(_in_channels, num_filters, kernel_size=kernel_size, stride=2, padding=padding,
                                   output_padding=output_padding, bias=False))
            self.features.append(nn.BatchNorm2d(num_filters))
            self.features.append(nn.ReLU(inplace=True))

        if with_bias_end:
            self.features.append(
                nn.Conv2d(num_filters, conv_num_filters, kernel_size=conv_kernel_size, padding=pad, bias=True))
        else:
            self.features.append(
                nn.Conv2d(num_filters, conv_num_filters, kernel_size=conv_kernel_size, padding=pad, bias=False))
            self.features.append(nn.BatchNorm2d(conv_num_filters))
            self.features.append(nn.ReLU(inplace=True))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                if with_bias_end:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)

    def forward(self, x):
        for i, l in enumerate(self.features):
            x = l(x)
        return x


# The backbone for resnet
class ResNetBackbone(nn.Module):

    def __init__(self, block, layers, in_channel=3):
        self.inplanes = 64
        super(ResNetBackbone, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, mean=0, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class ResnetNoStage(nn.Module):
    """
    The network takes image as input, output maps
    that can be interpreted as 2d heatmap, 2d heatmap + location map
    or 3d heatmap, depends on the value of config.depth_per_keypoint
    input: tensor in the size of (batch_size, config.image_channels, image_width, image_height)
    output: tensor in the size of (batch_size, confg.num_keypoints * config.depth_per_keypoint, image_width/4, image_height/4)
    """
    def __init__(self, config=ResnetNoStageConfig()):
        super(ResnetNoStage, self).__init__()
        block_type, layers, channels, name = resnet_spec[config.num_layers]
        self.backbone_net = ResNetBackbone(block_type, layers, config.image_channels)
        self.head_net = DeconvHead(
            channels[-1],
            config.num_deconv_layers,
            config.num_deconv_filters,
            config.num_deconv_kernel,
            config.final_conv_kernel,
            config.num_keypoints,
            config.depth_per_keypoint)

    def forward(self, x):
        x = self.backbone_net(x)
        x = self.head_net(x)
        return x


def initialize_backbone_from_modelzoo(
        backbone,  # type: ResNetBackbone,
        resnet_num_layers,  # type: int
        image_channels,  # type: int
    ):
    assert image_channels == 3 or image_channels == 4
    _, _, _, name = resnet_spec[resnet_num_layers]
    org_resnet = model_zoo.load_url(model_urls[name])
    # Drop orginal resnet fc layer, add 'None' in case of no fc layer, that will raise error
    org_resnet.pop('fc.weight', None)
    org_resnet.pop('fc.bias', None)
    # Load the backbone
    if image_channels is 3:
        backbone.load_state_dict(org_resnet)
    elif image_channels is 4:
        # Modify the first conv
        conv1_weight_old = org_resnet['conv1.weight']
        conv1_weight = torch.zeros((64, 4, 7, 7))
        conv1_weight[:, 0:3, :, :] = conv1_weight_old
        avg_weight = conv1_weight_old.mean(dim=1, keepdim=False)
        conv1_weight[:, 3, :, :] = avg_weight
        org_resnet['conv1.weight'] = conv1_weight
        # Load it
        backbone.load_state_dict(org_resnet)


def init_from_modelzoo(
        network,  # type: ResnetNoStage,
        config,  # type: ResnetNoStageConfig
    ):
    initialize_backbone_from_modelzoo(
        network.backbone_net,
        config.num_layers,
        config.image_channels)

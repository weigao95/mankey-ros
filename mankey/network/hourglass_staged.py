import torch
import torch.nn as nn
import torch.nn.functional as F
import attr


@attr.s
class HourglassNetConfig(object):
    """
    Config class for the staged hourglass network.
    Note that only the final stage is able to produce 3d predictions
    All other layers are only 2d heatmap for supervision only
    """
    image_channels = 3
    num_stages = 2
    num_blocks = 4
    # Need to modifiy the num_keypoints to match the required value
    num_keypoints = -1

    # For 2d keypoint, this value should be one
    # For 2d keypoint + depth, this value should be 2
    # For 3d volumetric keypoint, this value should be the depth resolution
    depth_per_keypoint = 1


class Bottleneck(nn.Module):
    """
    A customed bottlenet with expsansion 2
    """
    expansion = 2

    def __init__(self,
                 inplanes,  # type: int,
                 planes,  # type: int,
                 stride=1,
                 downsample=None):
        super(Bottleneck, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return out


class HourglassModule(nn.Module):
    """
    The basic hourglass module used in hourglass network
    """
    def __init__(self,
                 num_blks,  # type: int
                 planes,  # type: int
                 depth,  # type: int
    ):
        super(HourglassModule, self).__init__()
        self._depth = depth
        self._hg = self._make_hourglass(num_blks, planes, depth)

    @staticmethod
    def _make_residual(
            num_blks,  # type: int
            planes,  # type: int
    ):
        layers = []
        for i in range(0, num_blks):
            layers.append(Bottleneck(planes * Bottleneck.expansion, planes))
        return nn.Sequential(*layers)

    def _make_hourglass(
            self,
            num_blks,  # type: int
            planes,  # type: int
            depth,  # type: int
    ):
        hg = []
        for i in range(depth):
            res = []
            for j in range(3):
                res.append(self._make_residual(num_blks=num_blks, planes=planes))
            if i == 0:
                res.append(self._make_residual(num_blks=num_blks, planes=planes))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _hour_glass_forward(self, n, x):
        up1 = self._hg[n-1][0](x)
        low1 = F.max_pool2d(x, 2, stride=2)
        low1 = self._hg[n-1][1](low1)

        if n > 1:
            low2 = self._hour_glass_forward(n-1, low1)
        else:
            low2 = self._hg[n-1][3](low1)
        low3 = self._hg[n-1][2](low2)
        up2 = F.interpolate(low3, scale_factor=2)
        out = up1 + up2
        return out

    def forward(self, x):
        return self._hour_glass_forward(self._depth, x)


class HourglassNet(nn.Module):
    """
    The hourglass network for 3d keypoint detection.
    The network will produce intermediate heatmap for training, but
    only in 2D. Only the final heatmap can be provided in 3d, either
    in the form of location map or volumetric map.
    """
    def __init__(self, config=HourglassNetConfig()):
        super(HourglassNet, self).__init__()

        # The first encoding layers
        self.inplanes = 64
        self.num_feats = 128
        self.num_stacks = config.num_stages
        self.conv1 = nn.Conv2d(config.image_channels, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=True)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2, stride=2)

        # The following encoding
        self.layer1 = self._make_residual(self.inplanes, 1)
        self.layer2 = self._make_residual(self.inplanes, 1)
        self.layer3 = self._make_residual(self.num_feats, 1)

        # The hourglass module
        ch = self.num_feats * Bottleneck.expansion
        hg, res, fc, score, fc_, score_ = [], [], [], [], [], []
        for i in range(config.num_stages):
            hg.append(HourglassModule(config.num_blocks, self.num_feats, 4))
            res.append(self._make_residual(self.num_feats, config.num_blocks))
            fc.append(self._make_fc(ch, ch))
            if i is self.num_stacks - 1:
                score.append(nn.Conv2d(ch, config.num_keypoints * config.depth_per_keypoint, kernel_size=1, bias=True))
            else:
                score.append(nn.Conv2d(ch, config.num_keypoints, kernel_size=1, bias=True))
                fc_.append(nn.Conv2d(ch, ch, kernel_size=1, bias=True))
                score_.append(nn.Conv2d(config.num_keypoints, ch, kernel_size=1, bias=True))

        # Collect them
        self.hg = nn.ModuleList(hg)
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)
        self.score = nn.ModuleList(score)
        self.fc_ = nn.ModuleList(fc_)
        self.score_ = nn.ModuleList(score_)

        # Initialize the weight
        self._init_weight()

    def _make_residual(
            self,
            planes,  # type: int
            blocks,  # type: int
            stride=1
    ):
        downsample = None
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * Bottleneck.expansion,
                          kernel_size=1, stride=stride, bias=True),
            )

        layers = []
        layers.append(Bottleneck(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * Bottleneck.expansion
        for i in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _make_fc(
            self,
            inplanes,  # type: int
            outplanes,  # type: int
    ):
        bn = nn.BatchNorm2d(inplanes)
        conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=True)
        return nn.Sequential(
                conv,
                bn,
                self.relu,)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, mean=0, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        for i in range(self.num_stacks):
            y = self.hg[i](x)
            y = self.res[i](y)
            y = self.fc[i](y)
            score = self.score[i](y)
            out.append(score)
            if i < self.num_stacks - 1:
                fc_ = self.fc_[i](y)
                score_ = self.score_[i](score)
                x = x + fc_ + score_

        return out

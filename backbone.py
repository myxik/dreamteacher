import timm
import torch as T
import torch.nn as nn

from torch.nn import functional as F


class FPN(nn.Module):
    """
    always a list of feature maps comes and then list of same length is returned
    """
    def __init__(self, in_channels, out_channel, pooling_out_channels):
        super().__init__()
        self.conv_layers = [
            nn.Conv2d(in_channel, out_channel, 1, bias=False)
            for in_channel in in_channels
        ]
        self.conv_layers.reverse()
        self.conv_layers = nn.ModuleList(self.conv_layers)
        self.pooling_conv = nn.Conv2d(pooling_out_channels, out_channel, 1, bias=False)

    def forward(self, pooling_out, feature_maps):
        output = []
        upscaled_pooling = self.pooling_conv(pooling_out)
        feature_maps.reverse()

        for idx, feature_map in enumerate(feature_maps):
            upscaled_pooling = F.interpolate(upscaled_pooling, scale_factor=2, mode="bilinear", align_corners=True)
            convolved_feature_map = self.conv_layers[idx](feature_map)
            output.append(convolved_feature_map + upscaled_pooling)
        output.reverse()
        return output


class PPM(nn.Module):
    """
    pools a feature map to 4 different sizes, convolves and concats
    """
    def __init__(self, pool_sizes, num_channels):
        super().__init__()
        self.pool_sizes = pool_sizes
        num_poolings = len(self.pool_sizes)
        self.poolings = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_size),
                nn.Conv2d(num_channels, num_channels // num_poolings, 1, bias=False),
            )
            for pool_size in self.pool_sizes
        ])

    def forward(self, x):
        assert len(x.shape) == 4
        _, _, h, w = x.size()

        pools = [x]
        for pool in self.poolings:
            convolved = pool(x)
            pooling_out = F.interpolate(convolved, size=(h, w), mode="bilinear", align_corners=True)
            pools.append(pooling_out)

        return T.cat(pools, dim=1)
    

class ChannelBalancer(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()

        self.convs = nn.ModuleList([
            nn.Conv2d(channel_in, channel_out, 1, bias=False) if channel_in!=channel_out else nn.Identity()
            for channel_in, channel_out in zip(channels_in, channels_out)
        ])

    def forward(self, feature_maps):
        output = []
        for idx, feature_map in enumerate(feature_maps):
            output.append(self.convs[idx](feature_map))
        return output


class ImageBackbone(nn.Module):
    def __init__(self, channels_out, image_size):
        super().__init__()

        self.feature_extractor = timm.create_model(
            "resnet50.a2_in1k",
            pretrained=False,
            features_only=True
        )

        self.ppm = PPM([1,2,3,6], 2048)
        self.fpn = FPN([64, 256, 512, 1024], 1024, 4096)
        self.ch = ChannelBalancer([1024, 1024, 1024, 1024], channels_out)

    def forward(self, x):
        features = self.feature_extractor(x)

        ppm_out = self.ppm(features[-1])
        fpn_out = self.fpn(ppm_out, features[:-1])
        ch_out = self.ch(fpn_out)
        return ch_out

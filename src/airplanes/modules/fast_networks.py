from typing import List

import torch
import torch.nn as nn

from airplanes.modules.layers import ConvUpsampleAndConcatBlock


class SkipDecoder(nn.Module):
    def __init__(self, input_channels: List[int]):
        super(SkipDecoder, self).__init__()

        input_channels = input_channels[::-1]
        self.input_channels = input_channels
        self.output_channels = [256, 128, 64, 64]
        self.num_ch_dec = self.output_channels[::-1]

        self.block1 = ConvUpsampleAndConcatBlock(
            in_ch=input_channels[0],
            out_ch=self.output_channels[0],
            skip_chns=input_channels[1],
        )
        self.block2 = ConvUpsampleAndConcatBlock(
            in_ch=self.output_channels[0],
            out_ch=self.output_channels[1],
            skip_chns=self.input_channels[2],
        )
        self.block3 = ConvUpsampleAndConcatBlock(
            in_ch=self.output_channels[1],
            out_ch=self.output_channels[2],
            skip_chns=self.input_channels[3],
        )
        self.block4 = ConvUpsampleAndConcatBlock(
            in_ch=self.output_channels[2],
            out_ch=self.output_channels[3],
            skip_chns=self.input_channels[4],
        )

    def forward(self, features):
        output_features = {}
        x = features[-1]

        x = self.block1(x, features[-2])
        output_features[f"feature_s3_b1hw"] = x

        x = self.block2(x, features[-3])
        output_features[f"feature_s2_b1hw"] = x

        x = self.block3(x, features[-4])
        output_features[f"feature_s1_b1hw"] = x

        x = self.block4(x, features[-5])
        output_features[f"feature_s0_b1hw"] = x

        return output_features


class SkipDecoderRegression(SkipDecoder):
    def __init__(self, input_channels: List[int], num_output_channels: int):
        super(SkipDecoderRegression, self).__init__(input_channels)

        self.num_output_channels = num_output_channels

        self.out1 = nn.Sequential(
            nn.Conv2d(self.output_channels[0], 64, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(64, self.num_output_channels, kernel_size=1),
        )

        self.out2 = nn.Sequential(
            nn.Conv2d(self.output_channels[1], 64, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(64, self.num_output_channels, kernel_size=1),
        )

        self.out3 = nn.Sequential(
            nn.Conv2d(self.output_channels[2], 64, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(64, self.num_output_channels, kernel_size=1),
        )

        self.out4 = nn.Sequential(
            nn.Conv2d(self.output_channels[3], 64, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(64, self.num_output_channels, kernel_size=1),
        )

    def forward(self, features: List[torch.Tensor], return_features: bool = False):
        output_features = super(SkipDecoderRegression, self).forward(features)
        outputs = {}

        outputs[f"output_s3_b{self.num_output_channels}hw"] = self.out1(
            output_features[f"feature_s3_b1hw"]
        )
        outputs[f"output_s2_b{self.num_output_channels}hw"] = self.out2(
            output_features[f"feature_s2_b1hw"]
        )
        outputs[f"output_s1_b{self.num_output_channels}hw"] = self.out3(
            output_features[f"feature_s1_b1hw"]
        )
        outputs[f"output_s0_b{self.num_output_channels}hw"] = self.out4(
            output_features[f"feature_s0_b1hw"]
        )

        if return_features:
            return outputs, output_features
        else:
            return outputs

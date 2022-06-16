# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict

from layers import *


class NCFDepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(NCFDepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()


        #for i in range(4, -1, -1):

        i = 4
        #begin for loop
        # upconv_0
        num_ch_in = self.num_ch_enc[-1]
        num_ch_out = self.num_ch_dec[i]
        self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

        # upconv_1
        num_ch_in = self.num_ch_dec[i]
        num_ch_in += self.num_ch_enc[i - 1]
        num_ch_out = self.num_ch_dec[i]
        self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)
        # end for loop


        i = 3
        #begin for loop
        # upconv_0
        num_ch_in = self.num_ch_dec[i + 1]
        num_ch_out = self.num_ch_dec[i]
        self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

        # upconv_1
        num_ch_in = self.num_ch_dec[i]
        num_ch_in += self.num_ch_enc[i - 1]
        num_ch_out = self.num_ch_dec[i]
        self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)
        # end for loop


        i = 2
        #begin for loop
        # upconv_0
        num_ch_in = self.num_ch_dec[i + 1]
        num_ch_out = self.num_ch_dec[i]
        self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

        # upconv_1
        num_ch_in = self.num_ch_dec[i]
        num_ch_in += self.num_ch_enc[i - 1]
        num_ch_out = self.num_ch_dec[i]
        self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)
        # end for loop


        i = 1
        #begin for loop
        # upconv_0
        num_ch_in = self.num_ch_dec[i + 1]
        num_ch_out = self.num_ch_dec[i]
        self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

        # upconv_1
        num_ch_in = self.num_ch_dec[i]
        num_ch_in += self.num_ch_enc[i - 1]
        num_ch_out = self.num_ch_dec[i]
        self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)
        # end for loop


        i = 0
        #begin for loop
        # upconv_0
        num_ch_in = self.num_ch_dec[i + 1]
        num_ch_out = self.num_ch_dec[i]
        self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

        # upconv_1
        num_ch_in = self.num_ch_dec[i]
        num_ch_out = self.num_ch_dec[i]
        self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)
        # end for loop


        #for s in self.scales:
        self.convs[("dispconv", 0)] = Conv3x3(self.num_ch_dec[0], self.num_output_channels)
        self.convs[("dispconv", 1)] = Conv3x3(self.num_ch_dec[1], self.num_output_channels)
        self.convs[("dispconv", 2)] = Conv3x3(self.num_ch_dec[2], self.num_output_channels)
        self.convs[("dispconv", 3)] = Conv3x3(self.num_ch_dec[3], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        #switched from dict to list. tracer doesn't seem to understand dicts
        self.outputs = [None] * (max(self.scales)+1)

        # decoder
        x = input_features[-1]

        #for i in range(4, -1, -1):

        i = 4
        #begin for loop
        x = self.convs[("upconv", i, 0)](x)
        x = [upsample(x)]

        #if self.use_skips and i > 0:
        x += [input_features[i - 1]]

        x = torch.cat(x, 1)
        x = self.convs[("upconv", i, 1)](x)

        #if i in self.scales:
        #self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))
        #end for loop

        i = 3
        #begin for loop
        x = self.convs[("upconv", i, 0)](x)
        x = [upsample(x)]

        #if self.use_skips and i > 0:
        x += [input_features[i - 1]]

        x = torch.cat(x, 1)
        x = self.convs[("upconv", i, 1)](x)

        #if i in self.scales:
        self.outputs[i] = self.sigmoid(self.convs[("dispconv", i)](x))
        #end for loop

        i = 2
        #begin for loop
        x = self.convs[("upconv", i, 0)](x)
        x = [upsample(x)]

        #if self.use_skips and i > 0:
        x += [input_features[i - 1]]

        x = torch.cat(x, 1)
        x = self.convs[("upconv", i, 1)](x)

        #if i in self.scales:
        self.outputs[i] = self.sigmoid(self.convs[("dispconv", i)](x))
        #end for loop

        i = 1
        #begin for loop
        x = self.convs[("upconv", i, 0)](x)
        x = [upsample(x)]

        #if self.use_skips and i > 0:
        x += [input_features[i - 1]]

        x = torch.cat(x, 1)
        x = self.convs[("upconv", i, 1)](x)

        #if i in self.scales:
        self.outputs[i] = self.sigmoid(self.convs[("dispconv", i)](x))
        #end for loop

        i = 0
        #begin for loop
        x = self.convs[("upconv", i, 0)](x)
        x = [upsample(x)]

        #if self.use_skips and i > 0:
        #x += [input_features[i - 1]]

        x = torch.cat(x, 1)
        x = self.convs[("upconv", i, 1)](x)

        #if i in self.scales:
        self.outputs[i] = self.sigmoid(self.convs[("dispconv", i)](x))
        #end for loop

        return self.outputs
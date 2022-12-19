# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import PIL.Image as pil
import argparse
import glob
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import os
import sys
import torch
import torch.onnx
#from torch2trt import torch2trt
from torchsummary import summary
from torchvision import transforms, datasets

import networks
from evaluate_depth import STEREO_SCALE_FACTOR
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist

from onnxsim import simplify
import onnx
from onnx import version_converter, helper

import onnx_graphsurgeon as gs

def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple conversion for mondepth2 models.')

    parser.add_argument('--model_name', type=str,
                        help='name of a pretrained model to use',
                        choices=[
                            "mono_640x192",
                            "stereo_640x192",
                            "mono+stereo_640x192",
                            "mono_no_pt_640x192",
                            "stereo_no_pt_640x192",
                            "mono+stereo_no_pt_640x192",
                            "mono_1024x320",
                            "stereo_1024x320",
                            "mono+stereo_1024x320"])
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')

    return parser.parse_args()


def load_pretrained():
    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    download_model_if_doesnt_exist(args.model_name)
    model_path = os.path.join("models", args.model_name)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = networks.NCFDepthDecoder( #NCF - No Control Flow for onnx conversion
        num_ch_enc=encoder.num_ch_enc, scales=range(4) )

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval().cuda()

    return encoder, depth_decoder, feed_height, feed_width


# ACCEPTS MODEL_NAME, LOADED ENCODER AND DECODER ALONG WITH INPUT HEIGHT AND WIDTH AND EXPORTS ONNX MODEL
def convert_to_onnx(model_name, encoder, depth_decoder, height, width):

    """Function to convert to onnx
    """
    # CONVERT TO ONNX AND SAVE
    print("converting to onnx")

    with torch.no_grad():

        export_path = model_path = os.path.join("exports", model_name)

        #merge models
        #output index of 0 represents highest scale
        merged = networks.MergedModel(encoder, depth_decoder, 0) #MergedModels executes the second model with the output of the first

        #load input shapes
        encoder_input_shape = (1, 3, height, width ) #tuple(next(encoder.parameters()).size())

        # example data to feed conversion
        encoder_ones = torch.ones(encoder_input_shape).cuda().contiguous()

        # print("model summary: ")
        # summary(merged, encoder_input_shape)

        export_file = os.path.join(export_path, model_name + ".pth")
        torch.save(merged, export_file )

        #save to onnx
        export_file = os.path.join(export_path, model_name + ".onnx")

        torch.onnx.export(merged, encoder_ones, export_file, verbose=True, opset_version=11, input_names=["input"], output_names=["output"])

        print("finished onnx export")

        print("simplifying model")

        model = onnx.load(export_file)
        # convert model
        model_simp, check = simplify(model)

        assert check, "Simplified ONNX model could not be validated"


        onnx.save(model_simp, os.path.join(export_path, model_name + ".simplified.onnx"))

        print("saved simplified model")

        print("constant folding model")

        graph = gs.import_onnx(model_simp)

        graph.toposort()
        graph.fold_constants(error_ok=False).cleanup()

        onnx.save(gs.export_onnx(graph), os.path.join(export_path, model_name + ".simplified.gs.onnx"))

        graph

    print('-> Done!')


if __name__ == '__main__':
    args = parse_args()

    encoder, depth_decoder, feed_height, feed_width = load_pretrained()

    convert_to_onnx(args.model_name, encoder, depth_decoder, feed_height, feed_width)


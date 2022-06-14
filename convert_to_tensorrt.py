# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
import torch.onnx
from torchvision import transforms, datasets
from torchsummary import summary

import networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist
from evaluate_depth import STEREO_SCALE_FACTOR
from torch2trt import torch2trt


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



def convert_to_tensorrt(args):

    """Function to convert to tensorrt
    """

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
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4)) #TODO setting skips to false and commenting out the model loading (line 85) fixes one control flow error, but others are still present.

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval().cuda()


    # CONVERT TO TENSORRT AND SAVE
    print("converting to tensorrt")

    with torch.no_grad():

        #merge models
        merged = networks.MergedModel(encoder, depth_decoder) #MergedModels executes the second model with the output of the first

        #load input shapes
        encoder_input_shape = (1, 3, feed_height, feed_width) #tuple(next(encoder.parameters()).size())
        depth_decoder_input_shape = (256, 512, 3, 3)#tuple(next(depth_decoder.parameters()).size()) #TODO for some reason this is always inaccurate

        # example data to feed conversion
        encoder_ones = torch.ones(encoder_input_shape).cuda()
        decoder_ones = torch.ones(depth_decoder_input_shape).cuda()

        print("model input shape: ", encoder_ones.shape)


        # print("model summary: ")
        # summary(merged, encoder_input_shape) #TODO fails because of input shape

        #save to onnx
        #torch.onnx.export(encoder, encoder_ones, os.path.join(model_path, "encoder.onnx"), verbose=True)
        #torch.onnx.export(depth_decoder, decoder_ones, os.path.join(model_path, "depth.onnx"), verbose=False) #TODO this fails because the decoder doesn't work with the tracing conversion

        #torch.onnx.export(merged, encoder_ones, os.path.join(model_path, "merged.onnx"), verbose=False)

        #convert to tensorrt
        encoder_trt = torch2trt(encoder, [encoder_ones])
        # decoder_trt = torch2trt(depth_decoder, [decoder_ones) #TODO can't get the input shape right for some reason. Merged model is preferred.

        merged_trt = torch2trt(merged, [encoder_ones])


    print("saving engine file")

    merged_trt_path = os.path.join(model_path, "merged.engine")
    with open(merged_trt_path, "wb") as f:
        f.write(merged_trt.engine.serialize())


    print("Testing model")

    # TEST TENSORT MODEL

    # Load image and preprocess
    input_image = pil.open("assets/test_image.jpg").convert('RGB')
    original_width, original_height = input_image.size
    input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
    input_image = transforms.ToTensor()(input_image).unsqueeze(0)

    # prediction
    input_image = input_image.to(device)

    outputs = merged_trt(input_image)

    disp = outputs[0]
    disp_resized = torch.nn.functional.interpolate(
        disp, (original_height, original_width), mode="bilinear", align_corners=False)

    # Saving colormapped depth image
    disp_resized_np = disp_resized.squeeze().cpu().numpy()
    vmax = np.percentile(disp_resized_np, 95)
    normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
    im = pil.fromarray(colormapped_im)
    im.save("assets/test_disp.jpg")

    print('-> Done!')


if __name__ == '__main__':
    args = parse_args()
    convert_to_tensorrt(args)

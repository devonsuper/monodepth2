import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import argparse
import torch

import PIL.Image as pil
from torchvision import transforms, datasets
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm

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

args = parse_args()

assert args.model_name, "must choose model"

dims = args.model_name.split("_")[-1].split("x")
print(dims)
feed_height = int( dims[0] )
feed_width = int ( dims[1] )


model = tf.saved_model.load(tf_model_path)
model.trainable = False


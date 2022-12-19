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

import os
import glob

import cv2

def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple conversion for mondepth2 models.')
    parser.add_argument('--image_path', type=str,
                        help='path to a test image or folder of images', required=True)
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="jpg")
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

tflite_model_path = "exports/" + args.model_name + "/" + args.model_name + ".tflite"


interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()  # Needed before execution!

# FINDING INPUT IMAGES
if os.path.isfile(args.image_path):
    # Only testing on a single image
    paths = [args.image_path]
    output_directory = os.path.dirname(args.image_path)
elif os.path.isdir(args.image_path):
    # Searching folder for images
    paths = glob.glob(os.path.join(args.image_path, '*.{}'.format(args.ext)))
    output_directory = args.image_path
else:
    raise Exception("Can not find args.image_path: {}".format(args.image_path))

print("-> Predicting on {:d} test images".format(len(paths)))

for idx, image_path in enumerate(paths):

    if image_path.endswith("_disp.jpg"):
        # don't try to predict disparity for a disparity image!
        continue

    #input_image = pil.open("assets/test_image.jpg").convert('RGB')
    input_image = pil.open(image_path).convert('RGB')
    original_width, original_height = input_image.size
    #input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)

    img = cv2.imread(image_path)
    img = (cv2.resize(img, (feed_width, feed_height)).astype(np.float32).transpose(2,1,0)/255)#transpose((1,0,2))/255)


    output = interpreter.get_output_details()[0]  # Model has single output.
    input = interpreter.get_input_details()[0]  # Model has single input.

    #input_image = transforms.ToTensor()(input_image).unsqueeze(0).reshape(input["shape"]).numpy()
    #input_image = (input_image/255).astype(np.float32)


    print(input["shape"])
    print(img.shape)

    interpreter.set_tensor(input['index'], [img])
    interpreter.invoke()
    outputs = interpreter.get_tensor(output['index'])

    disp = torch.FloatTensor(outputs)
    disp_resized = disp #torch.nn.functional.interpolate(
    #     disp, (original_height, original_width), mode="bilinear", align_corners=False)

    # Saving colormapped depth image
    disp_resized_np = disp_resized.squeeze().cpu().numpy()
    vmax = np.percentile(disp_resized_np, 95)
    normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
    im = pil.fromarray(colormapped_im)

    output_name = os.path.splitext(os.path.basename(image_path))[0]
    name_dest_im = os.path.join(output_directory, "{}_disp.jpeg".format(output_name))
    im.save(name_dest_im)

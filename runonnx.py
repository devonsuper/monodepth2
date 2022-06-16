import PIL.Image as pil
import argparse
import glob
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import onnxruntime as ort
import os
import sys
import torch
import torch.onnx
from torchsummary import summary
from torchvision import transforms, datasets

import networks
from evaluate_depth import STEREO_SCALE_FACTOR
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist


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
    return parser.parse_args()


def main(args):

    device = torch.device("cuda")

    download_model_if_doesnt_exist(args.model_name)
    model_path = os.path.join("models", args.model_name)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = networks.NDDepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(
            4), )  # TODO setting skips to false and commenting out the model loading (line 85) fixes one control flow error, but others are still present.

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    #depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()

    # Load image and preprocess
    input_image = pil.open("assets/test_image.jpg").convert('RGB')
    original_width, original_height = input_image.size
    input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
    input_image = transforms.ToTensor()(input_image).unsqueeze(0)

    # prediction
    # input_image = input_image.to(device)
    # input_image = input_image.contiguous()  # from https://github.com/NVIDIA-AI-IOT/torch2trt/issues/220#issuecomment-569949961



    #run onnx

    ort_sess = ort.InferenceSession("exports/" + args.model_name + "/merged.onnx", providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
    outputs = ort_sess.run(None, {'input': input_image.numpy()})

    #test image

    disp = torch.FloatTensor(outputs[0])
    disp_resized = torch.nn.functional.interpolate(
        disp, (original_height, original_width), mode="bilinear", align_corners=False)

    # Saving colormapped depth image
    disp_resized_np = disp_resized.squeeze().cpu().numpy()
    vmax = np.percentile(disp_resized_np, 95)
    normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
    im = pil.fromarray(colormapped_im)
    im.save("assets/onnx_disp.jpg")

if(__name__ == "__main__"):
    args = parse_args()
    main(args)
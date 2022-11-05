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

tf_model_path = "exports/" + args.model_name + "/" + args.model_name + ".tensorflow"

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
tflite_model = converter.convert()



interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()  # Needed before execution!

input_image = pil.open("assets/test_image.jpg").convert('RGB')
original_width, original_height = input_image.size
input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)


input_image = transforms.ToTensor()(input_image).unsqueeze(0).transpose(2, 3).numpy()
#input_image = (input_image/255).astype(np.float32)

output = interpreter.get_output_details()[0]  # Model has single output.
input = interpreter.get_input_details()[0]  # Model has single input.

print(input["shape"])

interpreter.set_tensor(input['index'], input_image)
interpreter.invoke()
outputs = interpreter.get_tensor(output['index'])

disp = torch.FloatTensor(outputs)
disp_resized = torch.nn.functional.interpolate(
    disp, (original_height, original_width), mode="bilinear", align_corners=False)

# Saving colormapped depth image
disp_resized_np = disp_resized.squeeze().cpu().numpy()
vmax = np.percentile(disp_resized_np, 95)
normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
im = pil.fromarray(colormapped_im)
im.save("assets/tflite_disp.jpg")
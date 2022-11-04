import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import argparse
import torch

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

onnx_model_path = "exports/" + args.model_name + "/" + args.model_name + ".simplified.onnx"
print(onnx_model_path)

onnx_model = onnx.load(onnx_model_path)
tf_rep = prepare(onnx_model)

tf_model_path = "exports/" + args.model_name + "/" + args.model_name + ".tensorflow.pb"

tf_rep.export_graph(tf_model_path)

model = tf.saved_model.load(tf_model_path)
model.trainable = False

#input_tensor = tf.random.uniform([1, 3, feed_height, feed_width])
#out = model(**{'serving_default_input': input_tensor})

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
tflite_model = converter.convert()

tflite_model_path = "exports/" + args.model_name + "/" + args.model_name + ".tflite"

# Save the model
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)
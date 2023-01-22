import torch
from torch import nn
import export_onnx
import onnx
from onnx_tf.backend import prepare

import tensorflow as tf

import os
import numpy as np
import PIL.Image as pil

# to onnx
model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
model.eval()
# input_shape = (1, 3, 224, 224)
# input_ones = torch.ones(input_shape)
#
# # flatten_layer = nn.Flatten()
# # reshape_layer = nn.Unflatten(1, ( 3, 224, 224))
# #
# # model = nn.Sequential(flatten_layer, reshape_layer, model )
# output = model(input_ones)
#
#
# export_path = "models/mobilenetv2/model.pth"
#
# torch.save(model, export_path)
#
# print(model)
#
#
# print("input shape: ", input_shape)
#
# onnx_file_path = "exports/mobilenetv2/model.onnx"
#
# torch.onnx.export(model, input_ones, onnx_file_path, verbose=True, opset_version=11, input_names=["input"], output_names=["output"])
#
# # to tflite
# onnx_model = onnx.load(onnx_file_path)
# tf_rep = prepare(onnx_model)
#
# tf_model_path = "exports/mobilenetv2/model.tensorflow"
# tf_rep.export_graph(tf_model_path)
#
# for i in range(2):
#     model = tf.saved_model.load(tf_model_path)
#     model.trainable = False
#
#     tf.saved_model.save(model, tf_model_path, signatures=model.signatures)
#
# # input_tensor = tf.random.uniform([1, 3, feed_height, feed_width])
# # out = model(**{'serving_default_input': input_tensor})
#
# # Convert the model
# converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
# converter.target_spec.supported_types = [tf.float32]
# converter.experimental_new_converter = True
# # converter.experimental_enable_resource_variables = True
#
# converter.target_spec.supported_ops = [
#     tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
#     #tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
# ]
#
# tflite_model = converter.convert()
#
tflite_model_path = "exports/mobilenetv2/model.tflite"
#
# # Save the model
# with open(tflite_model_path, 'wb') as f:
#     f.write(tflite_model)


#testing

import urllib
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)

from PIL import Image
from torchvision import transforms

input_image = Image.open(filename)

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)#.permute(0, 3, 1, 2)

#Torch evaluation

with torch.no_grad():
    output = model(input_batch)
# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
#print(output[0])
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
probabilities = torch.nn.functional.softmax(output[0], dim=0)
#print(probabilities)

# Read the categories
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
# Show top categories per image
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())


import matplotlib.pyplot as plt

input_numpy = input_tensor.permute(1,2,0).numpy()
plt.imshow( (input_numpy * 255).astype(np.uint8) )
#plt.imshow( input_image )
plt.show()

#tflite evaluation

interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()  # Needed before execution!

input_image = pil.open(filename).convert('RGB')
original_width, original_height = input_image.size
#input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)

img = input_numpy.transpose(2,1,0) #transpose((1,0,2))/255)


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

probabilities = torch.nn.functional.softmax(disp[0], dim=0)
#print(probabilities)

print("tensorflow predictions: ")

# Read the categories
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
# Show top categories per image
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())
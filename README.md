# Monodepth2 ONNX export 

A tool used to export monodepth2 models to ONNX file format.

Please refer to the original monodepth2 paper and github repository to find the original work:

> **Digging into Self-Supervised Monocular Depth Prediction**
>
> [Clément Godard](http://www0.cs.ucl.ac.uk/staff/C.Godard/), [Oisin Mac Aodha](http://vision.caltech.edu/~macaodha/), [Michael Firman](http://www.michaelfirman.co.uk) and [Gabriel J. Brostow](http://www0.cs.ucl.ac.uk/staff/g.brostow/)
>
> [ICCV 2019 (arXiv pdf)](https://arxiv.org/abs/1806.01260)

[Monodepth2 repository](https://github.com/nianticlabs/monodepth2)



This code is for non-commercial use; please see the [license file](LICENSE) for terms.


## ⚙️ Setup

Please follow the setup described in the original mondepth2 repository

## Usage

Convert a monodepth2 model to onnx with:

```shell
python3 export_onnx.py --model_name=mono+stereo_640x192
```
This will export the onnx file for mono+stereo_640x192. All models are exported with only the first output.

All pretrained monodepth2 models are supported

Available models:
1. mono_640x192
2. stereo_640x192
3. mono+stereo_640x192
4. mono_no_pt_640x192
5. stereo_no_pt_640x192
6. mono+stereo_no_pt_640x192
7. mono_1024x320
8. stereo_1024x320
9. mono+stereo_1024x320

To use a custom model, refer to the  convert_to_onnx() function.

## 👩‍⚖️ License
Copyright © Niantic, Inc. 2019. Patent Pending.
All rights reserved.
Please see the [license file](LICENSE) for terms.

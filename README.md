# DNN Model Residual Encoding Framework for Fast Parameter Transfer

This repository is the official implementation of 《DNN Model Residual Encoding Framework for Fast Parameter Transfer》

<img src="figures/lossy.png">

## Requirements

To install requirements:

```requirements
pip install -r requirements.txt
```
## Setup

To compile entropy encoding and decoding algorithms:
```setup
cd codec
python setup.py build_ext --inplace
```

## Compress Demo

A demo for compress parameters:

```demo
python demo.py --pre-epoch 'weights/yolov5n/21_0.01.pt' --cur-epoch 'weights/yolov5n/24_0.01.pt' --dnn 'yolo' --method 'ResEntropy16bits'
```

## Evaluation

1. Train [ResNet-18](https://github.com/pytorch/examples/tree/main/imagenet) and [YOLOv5n](https://github.com/ultralytics/yolov5), and save parameters for every epoch (filename: 'epoch_learningrate.pt', e.g. '21_0.01.pt').
2. To evaluate our method, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 


# DNN Model Residual Encoding Framework for Fast Parameter Transfer

This repository is the official implementation of 《DNN Model Residual Encoding Framework for Fast Parameter Transfer》

## Effect on Peformance
ResNet-18 (ImageNet): Current Epoch=90, Previous Epoch=90-n
|Methods|Accuracy@Top1|Accuracy@Top5|
|-------|-------------|-------------|
|Original|69.546|89.166|
|Float16|69.536|89.184|
|ResidualFloat16 n=1|69.546|89.166|
|ResidualFloat16 n=3|69.544|89.166|
|ResidualFloat16 n=5|69.544|89.166|
|ResEntropy16bits n=1|69.546|89.166|
|ResEntropy16bits n=3|69.546|89.166|
|ResEntropy16bits n=5|69.546|89.166|

YOLOv5n (MicroSoft COCO): Current Epoch=200, Previous Epoch=200-n
|Methods|mAP@.5|mAP@.5:.95|
|-------|------|----------|
|Original|0.447|0.261|
|Float16|0.447|0.261|
|ResidualFloat16 n=1|0.447|0.261|
|ResidualFloat16 n=3|0.447|0.261|
|ResidualFloat16 n=5|0.447|0.261|
|ResEntropy16bits n=1|0.447|0.261|
|ResEntropy16bits n=3|0.447|0.261|
|ResEntropy16bits n=5|0.447|0.261|

## Requirements

To install requirements:

```requirements
pip install -r requirements.txt
```
## Setup

To compile entropy encoding and decoding algorithms:
```setup
cd DNN_param_encode/codec
python setup.py build_ext --inplace
```

## Compress Demo

A demo of compressing parameters in the latter epoch when parameters in two near epochs are given (the entropy encoding code is being optimized, and the speed is currently relatively slow):

```demo
cd DNN_param_encode
python demo.py --pre-epoch 'weights/yolov5n/21_0.01.pt' --cur-epoch 'weights/yolov5n/24_0.01.pt' --dnn 'yolo' --method 'ResEntropy16bits'
```

## Evaluation

1. Train [ResNet-18](https://github.com/pytorch/examples/tree/main/imagenet) and [YOLOv5n](https://github.com/ultralytics/yolov5), and save parameters for every epoch (filename: 'epoch_learningrate.pt', e.g. '21_0.01.pt').
2. To evaluate our method, run:

```eval
cd DNN_param_encode
python eval_lossless.py --learning-rate '0.01' --epoch-interval 3 --dnn 'yolo' --epoch-first 21 --epoch-last 100 --path-pt 'weights/yolov5n/'
or
python eval_lossy.py --method 'ResEntropy16bits' --learning-rate '0.01' --epoch-interval 3 --dnn 'yolo' --epoch-first 21 --epoch-last 100 --path-pt 'weights/yolov5n/'
```



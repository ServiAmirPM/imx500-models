# Raspberry Pi AI Camera (IMX500) Neural Network Models

Introducing the official Raspberry Pi AI Camera computer vision reference models 🚀 enabling real‑time vision AI applications at the edge!

The imx500-models repository is a [**collection of reference deep learning models**](https://github.com/ServiAmirPM/imx500-models/edit/main/README.md#reference-deep-learning-models) that are optimized 🪄 to run on the Raspberry Pi AI-Camera.

For each model you'll find an example code for executing and visualizing directly on a Raspberry Pi AI Camera, with just one execution command.

## Running Example Applications
The below examples are templates to help implement specific deep learning inference use-cases. These applications will help you get started with AI on Raspberry Pi, showcasing how to preprocess and postrpocess data for model inference and organize processing pipelines.

1. First, all reference deep learning models should be installed on Raspberry Pi OS with:

```
sudo apt install imx500-models
```
2. Then, make sure you are located on the [Picamera2 demo/example scripts](https://github.com/raspberrypi/picamera2/tree/main/examples/imx500) where the example scripts running the models are located.

```
git clone https://github.com/raspberrypi/picamera2.git

cd picamera2/examples/imx500
```

The example applications project is organized by application, for each example you will have to run a 1-line command (appear in the models tables in this page) indicating both model and **application name** per task, as appears below:
| Application Name                             | Task                  | Link   |
|----------------------------------------------|-----------------------|--------|
| `imx500_classification_demo.py`              | Classification        | [Link](https://github.com/raspberrypi/picamera2/blob/main/examples/imx500/imx500_classification_demo.py)|
| `imx500_object_detection_demo.py`            | Object Detection      | [Link](https://github.com/raspberrypi/picamera2/blob/main/examples/imx500/imx500_object_detection_demo.py)|
| `imx500_pose_estimation_higherhrnet_demo.py` | Pose Estimation       | [Link](https://github.com/raspberrypi/picamera2/blob/main/examples/imx500/imx500_pose_estimation_higherhrnet_demo.py)|
| `imx500_segmentation_demo.py`                | Segmentation          | [Link](https://github.com/raspberrypi/picamera2/blob/main/examples/imx500/imx500_segmentation_demo.py)|



## Reference Deep Learning Models
These models can be useful for quick deployment if you are interested in the categories that they were trained on.

### Classification
**Task:** Categorize input data into predefined classes and provide a confidence score.

**Training dataset:** _Imagenet_ - designed for use in visual object recognition research. It contains over 14 million images, making it one of the most extensive resources available for training deep learning models in computer vision tasks. It comprises 1000 classes.

| Model                              | Top 1 Accuracy (Float / Quantized)     | Input Resolution | Picamera2 example script                                                                                                            |
|------------------------------------|----------------------------------------|------------------|------------------------------------------------------------------------------------------------------------------------|
| efficientnet_b0                    | 73.876 / **72.128​**                    | 224x224          | imx500_classification_demo.py --model /usr/share/imx500-models/imx500_network_efficientnet_bo.rpk                      |
| efficientnet_lite0                 | 75.28 / **75.252**                     | 224x224          | imx500_classification_demo.py --model /usr/share/imx500-models/imx500_network_efficientnet_lite0.rpk                   |
| efficientnetv2_b0                  | 76.424 / **76.674​**                    | 224x224          | imx500_classification_demo.py --model /usr/share/imx500-models/imx500_network_efficientnetv2_b0.rpk                    |
| efficientnetv2_b1                  | 76.93 / **77.032​**                     | 240x240          | imx500_classification_demo.py --model /usr/share/imx500-models/imx500_network_efficientnetv2_b1.rpk                    |
| efficientnetv2_b2                  | 77.94 / **77.716**                     | 260x260          | imx500_classification_demo.py --model /usr/share/imx500-models/imx500_network_efficientnetv2_b2.rpk                    |
| levit_128s                         | 58.312 / **62.29​**                     | 224x224          | imx500_classification_demo.py --model /usr/share/imx500-models/imx500_network_levit_128s.rpk                           |
| mnasnet1.0                         | 73.078	/ **73.16​**                     | 224x224          | imx500_classification_demo.py --model /usr/share/imx500-models/imx500_network_mnasnet1.0.rpk                           |
| mobilenet_v2                       | 74.18 / **71.572​**                     | 224x224          | imx500_classification_demo.py --model /usr/share/imx500-models/imx500_network_mobilenet_v2.rpk                         |
| mobilevit_xs                       | 72.412 / **72.326​**                    | 256x256          | imx500_classification_demo.py --model /usr/share/imx500-models/imx500_network_mobilevit_xs.rpk                         |
| mobilevit_xxs                      | 67.40 / **67.44​0**                     | 256x256          | imx500_classification_demo.py --model /usr/share/imx500-models/imx500_network_mobilevit_xxs.rpk                        |
| regnetx_002                        | 68.20 / **68.352​**                     | 224x224          | imx500_classification_demo.py --model /usr/share/imx500-models/imx500_network_regnetx_002.rpk                          |
| regnety_002                        | 69.60 / **69.424​**                     | 224x224          | imx500_classification_demo.py --model /usr/share/imx500-models/imx500_network_regnety_002.rpk                          |
| regnety_004                        | 73.37 / **73.83​0**                     | 224x224          | imx500_classification_demo.py --model /usr/share/imx500-models/imx500_network_regnety_004.rpk                          |
| resnet18                           | 68.546 / **68.57​**                     | 224x224          | imx500_classification_demo.py --model /usr/share/imx500-models/imx500_network_resnet18.rpk                             |
| shufflenet_v2_x1_5                 | 72.498 / **72.194​**                    | 224x224          | imx500_classification_demo.py --model /usr/share/imx500-models/imx500_network_shufflenet_v2_x1_5.rpk                   |
| squeezenet1.0                      | 57.584 / **57.598**                    | 224x224          | imx500_classification_demo.py --model /usr/share/imx500-models/imx500_network_squeezenet1.0.rpk                        |  

### Object Detection
**Task:** Identify and locate multiple objects within an image by classifying each object.

**Training dataset**: _COCO_ - Designed to encourage research on a wide variety of object categories and is commonly used for benchmarking computer vision models. It is an essential dataset for researchers and developers working on object detection, segmentation, and pose estimation tasks. It comprises [80 classes](https://cocodataset.org/#explore).


| Model                              | mAP Accuracy (Float / Quantized)       | Input Resolution | Picamera2 example script                                                                                                            |
|------------------------------------|----------------------------------------|------------------|------------------------------------------------------------------------------------------------------------------------|
| efficientdet_lite0_pp              | 0.2518 / **0.252**​                     | 320x320          | imx500_object_detection_demo.py --model /usr/share/imx500-models/imx500_network_efficientdet_lite0_pp.rpk              |
| nanodet_plus_416x416               | 0.3316 / **0.332**​                     | 416x416          | imx500_object_detection_demo.py --model /usr/share/imx500-models/imx500_network_nanodet_plus_416x416.rpk               |
| nanodet_plus_416x416_pp            | 0.3232 / **0.32​0**​                     | 416x416          | imx500_object_detection_demo.py --model /usr/share/imx500-models/imx500_network_nanodet_plus_416x416_pp.rpk            |
| ssd_mobilenetv2_fpnlite_320x320_pp | 0.219 / **0.218** ​                     | 320x320          | imx500_object_detection_demo.py --model /usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk |

### Semantic Segmentation
**Task:** Assign a category to each pixel in an image, offering a comprehensive analysis of the image's content.

**Training dataset:** _PASCAL VOC_​ - Designed to encourage research on a wide variety of object categories and is commonly used for benchmarking computer vision models. It is an essential dataset for researchers and developers working on object detection, segmentation, and classification tasks.
It comprises [20 object categories](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/htmldoc/voc.html#SECTION00032000000000000000), including common objects like cars, bicycles, and animals, as well as more specific categories such as boats, sofas, and dining tables.

| Model                              | mIOU Accuracy (Float / Quantized)       | Input Resolution | Picamera2 example script                                                                                                            |
|------------------------------------|-----------------------------------------|------------------|------------|------------------------------------------------------------------------------------------------------------------------|
| deeplabv3plus                      | 0.724 / **0.7214​** ​                     | 320x320          | imx500_segmentation_demo.py  --model /usr/share/imx500-models/imx500_network_deeplabv3plus.rpk                         |

### Pose Estimation
**Task:** Detect key points or landmarks on objects or humans in images or videos.
**Training dataset:** _COCO (KeyPoints)_​

| Model                              | mAP Accuracy (Float / Quantized)       | Input Resolution | Picamera2 example script                                                                                                            |
|------------------------------------|----------------------------------------|------------------|------------|------------------------------------------------------------------------------------------------------------------------|
| higherhrnet_coco                   | 0.186762 / **0.188​**                   | 228x640          |            | imx500_pose_estimation_higherhrnet_demo.py --model /usr/share/imx500-models/imx500_network_higherhrnet_coco.rpk        |

## Licenses

Models in this repo are distributed under a number of licenses listed in in the [LICENSES](LICENSES) directory.

# Raspberry Pi AI Camera (IMX500) Neural Network Models

This repository contains neural network models designed to run on the Raspberry Pi AI Camera. These neural network models should be installed on Raspberry Pi OS with:

```console
sudo apt install imx500-models
```

## Demos

[Picamera2](https://github.com/raspberrypi/picamera2) demo/example scripts running these models are listed below:

| Model                              | Task                  | Input Size | Picamera2 example script                                                                                               |
|------------------------------------|-----------------------|------------|------------------------------------------------------------------------------------------------------------------------|
| efficientnet_bo                    | Classification        | 224x224    | imx500_classification_demo.py --model /usr/share/imx500-models/imx500_network_efficientnet_bo.rpk                      |
| efficientnet_lite0                 | Classification        | 224x224    | imx500_classification_demo.py --model /usr/share/imx500-models/imx500_network_efficientnet_lite0.rpk                   |
| efficientnetv2_b0                  | Classification        | 224x224    | imx500_classification_demo.py --model /usr/share/imx500-models/imx500_network_efficientnetv2_b0.rpk                    |
| efficientnetv2_b1                  | Classification        | 240x240    | imx500_classification_demo.py --model /usr/share/imx500-models/imx500_network_efficientnetv2_b1.rpk                    |
| efficientnetv2_b2                  | Classification        | 260x260    | imx500_classification_demo.py --model /usr/share/imx500-models/imx500_network_efficientnetv2_b2.rpk                    |
| levit_128s                         | Classification        | 224x224    | imx500_classification_demo.py --model /usr/share/imx500-models/imx500_network_levit_128s.rpk                           |
| mnasnet1.0                         | Classification        | 224x224    | imx500_classification_demo.py --model /usr/share/imx500-models/imx500_network_mnasnet1.0.rpk                           |
| mobilenet_v2                       | Classification        | 224x224    | imx500_classification_demo.py --model /usr/share/imx500-models/imx500_network_mobilenet_v2.rpk                         |
| mobilevit_xs                       | Classification        | 256x256    | imx500_classification_demo.py --model /usr/share/imx500-models/imx500_network_mobilevit_xs.rpk                         |
| mobilevit_xxs                      | Classification        | 256x256    | imx500_classification_demo.py --model /usr/share/imx500-models/imx500_network_mobilevit_xxs.rpk                        |
| regnetx_002                        | Classification        | 224x224    | imx500_classification_demo.py --model /usr/share/imx500-models/imx500_network_regnetx_002.rpk                          |
| regnety_002                        | Classification        | 224x224    | imx500_classification_demo.py --model /usr/share/imx500-models/imx500_network_regnety_002.rpk                          |
| regnety_004                        | Classification        | 224x224    | imx500_classification_demo.py --model /usr/share/imx500-models/imx500_network_regnety_004.rpk                          |
| resnet18                           | Classification        | 224x224    | imx500_classification_demo.py --model /usr/share/imx500-models/imx500_network_resnet18.rpk                             |
| shufflenet_v2_x1_5                 | Classification        | 224x224    | imx500_classification_demo.py --model /usr/share/imx500-models/imx500_network_shufflenet_v2_x1_5.rpk                   |
| squeezenet1.0                      | Classification        | 224x224    | imx500_classification_demo.py --model /usr/share/imx500-models/imx500_network_squeezenet1.0.rpk                        |  
| efficientdet_lite0_pp              | Object Detection      | 320x320    | imx500_object_detection_demo.py --model /usr/share/imx500-models/imx500_network_efficientdet_lite0_pp.rpk              |
| nanodet_plus_416x416               | object detection      | 416x416    | imx500_object_detection_demo.py --model /usr/share/imx500-models/imx500_network_nanodet_plus_416x416.rpk               |
| nanodet_plus_416x416_pp            | Object Detection      | 416x416    | imx500_object_detection_demo.py --model /usr/share/imx500-models/imx500_network_nanodet_plus_416x416_pp.rpk            |
| yolov8n_pp                         | object detection      | 640x640    | imx500_object_detection_demo.py --model /usr/share/imx500-models/imx500_network_yolov8n_pp.rpk                         |
| ssd_mobilenetv2_fpnlite_320x320_pp | Object Detection      | 320x320    | imx500_object_detection_demo.py --model /usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk |
| deeplabv3plus                      | Image Segmentation    | 320x320    | imx500_segmentation_demo.py  --model /usr/share/imx500-models/imx500_network_deeplabv3plus.rpk                         |
| higherhrnet_coco                   | Pose Estimation       | 228x640    | imx500_pose_estimation_higherhrnet_demo.py --model /usr/share/imx500-models/imx500_network_higherhrnet_coco.rpk        |

## Licenses

Models in this repo are distributed under a number of licenses listed in in the [LICENSES](LICENSES) directory.

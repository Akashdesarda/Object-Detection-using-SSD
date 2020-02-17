# Objet Detection using Single Shot Multibox Detector (SSD)

---

This is a Keras port of the SSD model architecture introduced by Wei Liu et al. in the paper [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) and originally implemented by Pierluigi Ferrari & can be found [here](https://github.com/pierluigiferrari/ssd_keras). This implementation is focussed towards two important points (which were missing in originall implementation):

1. Training and inference can be done on mediocre discreate local system GPUs (like I have used Nvidia gtx 1660 on my local setup).

2. A universal note book to perform all task:- training, training on custom dataset, inference on video or images, etc.

## APIS and Usage

---

## Training API

> This apis will be need to train any custom dataset. (Though some are common in both training and inference)

### limit_gpu: bool, by default True

Enable Tensorflow's limiting memory graph. This will need if the job is performed on local setup.
True: To enable
False: To disable

### mode: str

 What job to perform, either 'training' or 'inference'

### img_height: int, by default 300

Height of image

### img_width:int, by default 300

Width of image

### img_channels: int, by default 3

Color channel, 3 for RGB

> **_Loading annotation_**

### annotation_type: str, by default 'csv'

Format of annotation, either 'csv' or 'xml'

### train_load_images_into_memory: bool, by default True

True:Will load all images into memory; False: Keeeps on disk, but much slower

### validation_load_images_into_memory: bool, by default True

same as above

>**_Dataset location_**

### train_img_dir: str

csv or path containing training data

### train_image_set_filename:str

To be used only in coco data set.

### val_img_dir:str

validation data

### val_annotation_dir:str

vidation data annotation

### val_image_set_filename:str

To used only in coco dataset.

### classes:list

list of all classes

### n_classes no of classes: int
no of classes

> **_Training Hyperparameters_**

### l2_regularization:float, by default 0.5

L2 regularization penalizing factor

### pos_iou_threshold:float, by default 0.5

IOU threshold used for localization

### learning_rate: float, by default 0.001

Learning rate to train

### steps_per_epoch:int

no of steps per epoch to take

### batch_size: int

size of batch of data to train

### epochs:int

no of epoch to train on

>**_Saving training assets_**

### weight_save_path: str

path to save weights

### csv_log_save_path: str

path to save training job monitor csv

## Inference API

> This apis will be used for inference job

### limit_gpu: bool, by default True

Enable Tensorflow's limiting memory graph. This will need if the job is performed on local setup.
True: To enable
False: To disable

### mode: str

 What job to perform, either 'training' or 'inference'

### img_height: int, by default 300

Height of image

### img_width:int, by default 300

Width of image

### img_channels: int, by default 3

Color channel, 3 for RGB

### weights_path: str

path to load trained weights

### confidence_threshold:float, by default 0.5

Threshold to select prediction

>**_Saving Inference Assets_**

### predicted_frames_export_path:str

frames with predicted bounding box will be saved here

### video_output_path:str

video with predicted bounding box will be saved here

## Technology Used

---

### Core Technology:

Python, Keras (Tensorflow:114),OpenCV, FFmpeg, Nvidia cuda

### Tools:

MLflow (tracking experiments), DVC(version control data), Git(version control project)

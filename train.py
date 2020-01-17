import warnings
warnings.filterwarnings('ignore')
from typing import Dict
from timeit import default_timer as timer
from datetime import datetime
now = datetime.now().strftime("%d-%m-%Y")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TerminateOnNaN, CSVLogger
from keras import backend as K
from keras.models import load_model
from math import ceil
import numpy as np
from matplotlib import pyplot as plt

from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization

from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms

def limit_gpu(config: Dict):
    """Used to limit GPU usage. Based on Tensorflow's Logical limiting memory graph
    
    Parameters
    ----------
    config : Dict
        Config yaml/json containing all parameter
    """
    if config['limit_gpu'] is not False:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)

# def hype_prep(config: Dict):
#     """Set all general non-training parameter
    
#     Parameters
#     ----------
#     config : Dict
#         Config yaml/json containing all parameter
    
#     Returns
#     -------
#         img_height, img_width, img_channels, mean_color, swap_channels, n_classes,
#         scales, aspect_ratios, two_boxes_for_ar1, steps, offsets, clip_boxes,
#         variances, normalize_coords
#     """
# img_height = config['training']['img_height'] # Height  input images
# img_width = config['training']['img_width'] # Width  input images
# img_channels = config['training']['img_channels'] # Number of color channels 
mean_color = [123, 117, 104] # The per-channel mean of the images in the dataset. Do not change this value if you're using any of the pre-trained weights.
swap_channels = [2, 1, 0] # The color channel order in the original SSD is BGR, so we'll have the model reverse the color channel order of the input images.
scales_pascal = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05] # The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets
scales_coco = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05] # The anchor box scaling factors used in the original SSD300 for the MS COCO datasets
scales = scales_pascal
aspect_ratios = [[1.0, 2.0, 0.5],
                [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                [1.0, 2.0, 0.5],
                [1.0, 2.0, 0.5]] # The anchor box aspect ratios used in the original SSD300; the order matters
two_boxes_for_ar1 = True
steps = [8, 16, 32, 64, 100, 300] # The space between two adjacent anchor box center points for each predictor layer.
offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5] # The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.
clip_boxes = False # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
variances = [0.1, 0.1, 0.2, 0.2] # The variances by which the encoded target coordinates are divided as in the original implementation
normalize_coords = True
    # return img_height, img_width, img_channels, mean_color, swap_channels, n_classes, scales, aspect_ratios, two_boxes_for_ar1, steps, offsets, clip_boxes, variances, normalize_coords
           
def data_generator_func(config: Dict):
    """Data Generator for training data and validation data
    
    Parameters
    ----------
    config : Dict
        Config yaml/json containing all parameter
    
    Returns
    -------
        train_dataset, val_dataset
    """
        # Init DataGenerator
    start_data = timer()
    train_dataset = DataGenerator(load_images_into_memory=config['training']['train_load_images_into_memory'],
                                  hdf5_dataset_path=None)
    val_dataset = DataGenerator(load_images_into_memory=config['training']['validation_load_images_into_memory'],
                                 hdf5_dataset_path=None)
    if config['training']['train_load_images_into_memory'] is not False:
        print ("[INFO]... You have chosen to load data into memory")
    else:
        print("[WARNING]... You have chosen not to load data into memory. It will still work but will be much slower")
    
    train_img_dir = config['training']['train_img_dir']
    val_img_dir =  config['training']['val_img_dir']

    train_annotation_dir = config['training']['train_annotation_dir']
    val_annotation_dir =  config['training']['val_annotation_dir']

    train_image_set_filename = config['training']['train_image_set_filename']
    val_image_set_filename = config['training']['val_image_set_filename']

    classes = config['training']['classes']

    train_dataset.parse_xml(images_dirs=[train_img_dir],
                            image_set_filenames=[train_image_set_filename],
                            annotations_dirs=[train_annotation_dir],
                            classes=classes,
                            include_classes='all',
                            exclude_truncated=False,
                            exclude_difficult=False,
                            ret=False)

    val_dataset.parse_xml(images_dirs=[val_img_dir],
                        image_set_filenames=[val_image_set_filename],
                        annotations_dirs=[val_annotation_dir],
                        classes=classes,
                        include_classes='all',
                        exclude_truncated=False,
                        exclude_difficult=True,
                        ret=False)
    end_data = timer()
    print(f"[INFO]...Time taken by Data loading/transformation Job is {(end_data - start_data)/60:.2f} min(s)")
    return train_dataset, val_dataset
        
def callbacks(config: Dict):
    """Keras callbacks
    
    Parameters
    ----------
    config : Dict
        Config yaml/json containing all parameter
    
    Returns
    -------
    list
        callback_list
    """
    
    def lr_schedule(epoch):
        if epoch < 80:
            return 0.001
        elif epoch < 100:
            return 0.0001
        else:
            return 0.00001
    
    model_checkpoint = ModelCheckpoint(filepath=f"{config['training']['weight_save_path']}/"+"ssd_epoch-{epoch:02d}_loss-{loss:.4f}.hfd5",
                                        monitor='val_loss', 
                                        verbose=1,
                                        save_best_only=True)

    csv_logger = CSVLogger(filename=f"{config['training']['csv_log_save_path']}/logs_{now}.cvs", append=True)

    lr_schedular = LearningRateScheduler(lr_schedule, verbose=1)

    terminate_on_nan = TerminateOnNaN()

    callbacks_list = [model_checkpoint, csv_logger, lr_schedular, terminate_on_nan]
    
def ssd_model(config: Dict, train_dataset, val_dataset, callbacks_list):
    """Training SSD model
    
    Parameters
    ----------
    config : Dict
        Config yaml/json containing all parameter
    """
    start_train = timer()
    
    img_height = config['training']['img_height'] # Height  input images
    img_width = config['training']['img_width'] # Width  input images
    img_channels = config['training']['img_channels'] # Number of color channels
    n_classes = config['training']['n_classes'] # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
     
    model = ssd_300(image_size=(img_height, img_width, img_channels),
                n_classes=n_classes,
                mode='training',
                l2_regularization=config['training']['l2_regularization'],
                scales=scales,
                aspect_ratios_per_layer=aspect_ratios,
                two_boxes_for_ar1=two_boxes_for_ar1,
                steps=steps,
                offsets=offsets,
                clip_boxes=clip_boxes,
                variances=variances,
                normalize_coords=normalize_coords,
                subtract_mean=mean_color,
                swap_channels=swap_channels)
    
    weights_path = './weights/VGG_ILSVRC_16_layers_fc_reduced.h5'
    model.load_weights(weights_path, by_name=True)

    adam = Adam(lr=config['training']['learning_rate'], beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

    model.compile(optimizer=adam, loss=ssd_loss.compute_loss)
    
    batch_size = config['training']['batch_size']

    ssd_data_augmentation = SSDDataAugmentation(img_height=img_height,
                                                img_width=img_width,
                                                background=mean_color)

    convert_to_3_channels = ConvertTo3Channels()
    resize = Resize(height=img_height, width=img_width)

    predictor_sizes = [model.get_layer('conv4_3_norm_mbox_conf').output_shape[1:3],
                    model.get_layer('fc7_mbox_conf').output_shape[1:3],
                    model.get_layer('conv6_2_mbox_conf').output_shape[1:3],
                    model.get_layer('conv7_2_mbox_conf').output_shape[1:3],
                    model.get_layer('conv8_2_mbox_conf').output_shape[1:3],
                    model.get_layer('conv9_2_mbox_conf').output_shape[1:3]]

    ssd_input_encoder = SSDInputEncoder(img_height=img_height,
                                        img_width=img_width,
                                        n_classes=n_classes,
                                        predictor_sizes=predictor_sizes,
                                        scales=scales,
                                        aspect_ratios_per_layer=aspect_ratios,
                                        two_boxes_for_ar1=two_boxes_for_ar1,
                                        steps=steps,
                                        offsets=offsets,
                                        clip_boxes=clip_boxes,
                                        variances=variances,
                                        matching_type='multi',
                                        pos_iou_threshold=0.5,
                                        neg_iou_limit=0.5,
                                        normalize_coords=normalize_coords)
    
    train_generator = train_dataset.generate(batch_size=batch_size,
                                         shuffle=True,
                                         transformations=[ssd_data_augmentation],
                                         label_encoder=ssd_input_encoder,
                                         returns={'processed_images',
                                                  'encoded_labels'},
                                         keep_images_without_gt=False)

    val_generator = val_dataset.generate(batch_size=batch_size,
                                        shuffle=False,
                                        transformations=[convert_to_3_channels,
                                                        resize],
                                        label_encoder=ssd_input_encoder,
                                        returns={'processed_images',
                                                'encoded_labels'},
                                        keep_images_without_gt=False)

    # Get the number of samples in the training and validations datasets.
    train_dataset_size = train_dataset.get_dataset_size()
    val_dataset_size   = val_dataset.get_dataset_size()

    print(f"[INFO]...Number of images in the training dataset: {train_dataset_size}")
    print(f"[INFO]...Number of images in the validation dataset: {val_dataset_size}")
    print(f"[INFO]...Weights will be saved at {config['training']['weight_save_path']}")
    history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=config['training']['steps_per_epoch'],
                              epochs=config['training']['epochs'],
                              callbacks=callbacks_list,
                              validation_data=val_generator,
                              validation_steps=ceil(val_dataset_size/batch_size))
    end_train = timer()
    print(f"[INFO]...Total time taken by Training Job is {(end_train - start_train)/60:.2f} min(s)")
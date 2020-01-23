import warnings
warnings.filterwarnings('ignore')
import glob
from typing import Dict
from timeit import default_timer as timer
from datetime import datetime
now = datetime.now().strftime("%d-%m-%Y")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from keras import backend as K
from keras.preprocessing import image
from keras.models import load_model, model_from_json

from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization

from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms

from misc_utils.VideoProcessing import VideoProcessing

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
                
def detect_from_video(config: Dict):
    """Inference on a video with output a video showing all prediction
    
    Parameters
    ----------
    config : Dict
        Config yaml/json containing all parameter
    """
    video = config['inference']['video_input']['video_input_path']
    vp = VideoProcessing(video=video)
    vp.generate_frames(export_path=config['inference']['video_input']['video_to_frames_export_path'])
    if config['inference']['video_input']['video_to_frames_export_path'] == config['inference']['predicted_frames_export_path']:
        print("[Warning]... You have given Video to frame path same as prediction output path /nPredicted output will overwrite video to frame")
    img_height = config['inference']['img_height']
    img_width = config['inference']['img_width']
    model = ssd_300(image_size=(img_height, img_width, 3),
                n_classes=config['inference']['n_classes'],
                mode='inference',
                l2_regularization=0.0005,
                scales=[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05], # The scales for MS COCO are [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
                aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5]],
                two_boxes_for_ar1=True,
                steps=[8, 16, 32, 64, 100, 300],
                offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                clip_boxes=False,
                variances=[0.1, 0.1, 0.2, 0.2],
                normalize_coords=True,
                subtract_mean=[123, 117, 104],
                swap_channels=[2, 1, 0],
                confidence_thresh=0.5,
                iou_threshold=0.45,
                top_k=200,
                nms_max_output_size=400)

    # Load the trained weights into the model.
    weights_path = config['inference']['weights_path']

    model.load_weights(weights_path, by_name=True)
    
    # Working with image
    all_images = glob.glob(f"{config['inference']['video_input']['video_to_frames_export_path']}/*/*")
    
    # Setting Up Prediction Threshold
    confidence_threshold = config['inference']['confidence_threshold']
    
    # Setting Up Classes (Note Should be in same order as in training)
    classes = config['inference']['classes']
    
    vp.existsFolder(f"{config['inference']['predicted_frames_export_path']}/{video.split('.')[0]}")
    # Working with image
    for current_img in tqdm(all_images):
        current_img_name = current_img.split('/')[-1]
        orig_image = cv2.imread(current_img)
        input_images = [] # Store resized versions of the images here
        img = image.load_img(current_img, target_size=(img_height, img_width))
        img = image.img_to_array(img) 
        input_images.append(img)
        input_images = np.array(input_images)
        
        # Prediction
        y_pred = model.predict(input_images)

        # Using threshold
        y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]
        
        # Drawing Boxes
        for box in y_pred_thresh[0]:
            xmin = box[2] * orig_image.shape[1] / img_width
            ymin = box[3] * orig_image.shape[0] / img_height
            xmax = box[4] * orig_image.shape[1] / img_width
            ymax = box[5] * orig_image.shape[0] / img_height
            
            label = f"{classes[int(box[0])]}: {box[1]:.2f}"
            cv2.rectangle(orig_image, (int(xmin), int(ymin)),  (int(xmax),int(ymax)), (255, 0, 0), 2)
            cv2.putText(orig_image, label, (int(xmin), int(ymin)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imwrite(f"{config['inference']['predicted_frames_export_path']}/{video.split('.')[0]}/{current_img_name}", orig_image)
        
        # Creating video
    vp.generate_video(import_path=config['inference']['predicted_frames_export_path'],
                      export_path=config['inference']['video_input']['video_output_path'])
        
        
                    
        

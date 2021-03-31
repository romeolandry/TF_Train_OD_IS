import argparse
import os
import sys
import click
import tensorflow as tf

from configs.run_config import *
from scripts.Evaluation.utils import *
from scripts.Evaluation.models  import Model
from scripts.Evaluation.mask import Inference

if USE_GPU:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # Enable GPU dynamic memory allocation
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        try:
            if GPU_MEM_CAP is None:
                tf.config.experimental.set_memory_growth(gpu, True)
            else:
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(
                        memory_limit=GPU_MEM_CAP)])
        
        except RuntimeError as e:
            print('Can not set GPU memory config', e)
else:
    # Set CPU as available physical device
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
    tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')  



parser = argparse.ArgumentParser(description="Inference model. saved model or converted model")

parser.add_argument("--freezed", default=False, action="store_true",
    help=" Use freezed graph")

parser.add_argument("-m","--model", required=True ,
    help="the path to model directory(if image path_th to saved_model or freezed model")

parser.add_argument("--webcam", default=False, action="store_true",
    help=" Use web cam for inference")

parser.add_argument("--cam_input", default=camere_input, type= int,
    help="Index of availabe wedcam default 0")

parser.add_argument("--cam_width",
                    type= int,
                    default=camera_width,
                    help="camera width")

parser.add_argument("--cam_height", default=camera_height,type= int,
    help="camera height")

parser.add_argument("-p","--path_to_images",
    help="the path to directory of images or image path")

parser.add_argument("-l","--label", default=PATH_TO_LABELS_MAP ,
    help="the to label mab corresponding to model")

parser.add_argument("--th", default=.5 ,type= float,
    help=" Threshold for bounding box")

def main(args):
    if args.freezed:
        raise("Not available now use please savedModel")
    
    if not args.webcam:        
        if not args.path_to_images:
            raise("give the path for image file")
    
    threshold = float(args.th)
    model = Model(args.model)

    detection_model, model_name = model.Load_savedModel_model()
        
    infer = Inference(path_to_images=args.path_to_images,
                    path_to_labels=args.label,
                    model=detection_model,
                    model_name=model_name,
                    threshold=threshold)
    if args.webcam:
        camera_input = args.cam_input
        camera_width = args.cam_input
        camera_height = args.cam_height

        click.echo(click.style(f"\n Start inference using webcam \n", bold=True, fg='green'))

        if args.freezed:      
            infer.mask_inference_webcam_freezed_model(camera_input,
                                                    camera_width,
                                                    camera_height)
        else:

            infer.mask_inference_webcam_2(camera_input,
                                        camera_width,
                                        camera_height)
    else:

        click.echo(click.style(f"\n Start inference from {args.path_to_images} ... \n", bold=True, fg='green'))    

        infer.mask_inference_image_cv()
       

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

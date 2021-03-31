import argparse
import os
import sys
import click
import tensorflow as tf


from configs.run_config import *
from scripts.Evaluation.utils import *
from scripts.Evaluation.models  import Model
from scripts.Evaluation.ssd import Evaluation

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
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


parser = argparse.ArgumentParser(description="Eval model. saved model or converted model")


parser.add_argument("-m","--model", required=True ,
    help="the path to model directory(if image path_th to saved_model for tf or trt")

parser.add_argument("-p","--path_to_images", default=PATH_IMAGES +'/val2017',
    help="the path to directory of images or image path")

parser.add_argument("-a","--annotation", default=PATH_ANNOTATIONS +'/instances_val2017.json' ,
    help="the to label mab corresponding to model")

parser.add_argument("-b","--batch_size", type= int,
    default=32 ,
    help=" images pro Batch")

parser.add_argument("-s","--score_threshold", type= float,
    default=SCORE_THRESHOLD ,
    help="min score to considerate as prediction")

parser.add_argument("--iou", type= float,
    default=IOU_THRESHOLD ,
    help=" IoU to differentiate True positive and false positive prediction")

parser.add_argument("--data_size", type= float,
    default=DATA_SIZE_VALIDATION ,
    help=" Percentage of validation data to use")


def main(args):
    # image size for input model
    batch_size = args.batch_size
    model = Model(args.model)

    # Load model from saved model .pb
    detection_model, model_name = model.Load_savedModel_model()
    #detection_model, model_name = model.load_saved_model_for_inference()
        
    eval_ssd = Evaluation(path_to_images=args.path_to_images,
                       model=detection_model,
                       model_name=model_name,
                       path_to_annotations=args.annotation,
                       batch_size=batch_size,
                       score_threshold=args.score_threshold,
                       iou_threshold=args.iou,
                       validation_split=args.data_size)

    click.echo(click.style(f"\n Start Evaluation \n", bold=True, fg='green'))    

    results_coco, evaluatedIds, results_IoU = eval_ssd.generate_results_ssd_compute_map()
    eval_ssd.COCO_process_mAP(results_coco, evaluatedIds)
    eval_ssd.mAP_without_COCO_API(results_IoU, per_class=True)
    eval_ssd.mAP_without_COCO_API(results_IoU, per_class=False)
           

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

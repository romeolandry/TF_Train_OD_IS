import argparse
import os
import sys

sys.path.append(os.path.abspath(os.curdir))

from configs.run_config import *
from scripts.run.model  import Model
from scripts.Evaluation.metrics import *


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


parser = argparse.ArgumentParser(description="Evaluate an saved model")


parser.add_argument("-m","--model", required = True,
    help=" Use web cam for inference")

parser.add_argument("-b","--batch_size", default=32,
    help=" Use web cam for inference")

parser.add_argument("--path_to_images", default=PATH_IMAGES +'/val2017' ,
    help="the path to directory of images or image path")

parser.add_argument("--path_to_ann", default=PATH_ANNOTATIONS +'/val2017' ,
    help="the path to annotation file")

parser.add_argument("--ckpt", default='check-0' ,
    help="the path to checkpoint. if the evaluation will be proceed throught checkpoint")


if __name__ == "__main__":
    args = parser.parse_args()
    model = Model(args.model,args.ckpt)
                
    detection_model, model_name = model.Load_model()

    evaluate = Evaluation(args.path_to_images,
                      detection_model,
                      model_name,
                      args.path_to_ann,
                      args.batch_size)

    evaluate.generate_detecttion_results()
    evaluate.COCO_mAP_bbox()

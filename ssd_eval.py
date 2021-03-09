import argparse
import os
import sys
import click



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


from configs.run_config import *
from scripts.Evaluation.utils import *
from scripts.Evaluation.models  import Model
from scripts.Evaluation.ssd import Evaluation


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

parser.add_argument("-s","--input_size", type=int,
    default=640,
    help=" Input model size")


def main(args):
    # image size for input model
    batch_size = args.batch_size
    model = Model(args.model)
    input_size = args.input_size

    # Load model from saved model .pb
    detection_model, model_name = model.Load_savedModel_model()
    #detection_model, model_name = model.load_saved_model_for_inference()
        
    eval_ssd = Evaluation(path_to_images=args.path_to_images,
                       model=detection_model,
                       model_name=model_name,
                       path_to_annotations=args.annotation,
                       batch_size=batch_size,
                       input_size= input_size)

    click.echo(click.style(f"\n Start Evaluation \n", bold=True, fg='green'))    

    eval_ssd.generate_results_ssd_compute_map()
    #eval_ssd.validate_model_coco()    
           

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

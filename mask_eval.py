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
from scripts.Evaluation.mask import Evaluation


parser = argparse.ArgumentParser(description="Eval model. saved model or converted model")


parser.add_argument("-m","--model", required=True ,
    help="the path to model directory(if image path_th to saved_model or freezed model")


parser.add_argument("-s","--size",
                    help=" model input size ")


parser.add_argument("-p","--path_to_images", default=PATH_IMAGES +'/val2017',
    help="the path to directory of images or image path")

parser.add_argument("-a","--annotation", default=PATH_ANNOTATIONS +'/instances_val2017.json' ,
    help="the to label mab corresponding to model")

parser.add_argument("--batch_size", default=32 ,
    help=" images pro Batch")


def main(args):
    # image size for input model
    if not args.size:
        raise("specify the size input of your model")
    
    size = int(args.size)
    input_model_size = [size, size]
    batch_size = int(args.batch_size)
    model = Model(args.model)

    # Load model from saved model .pb
    detection_model, model_name = model.Load_savedModel_model()
        
    eval_mask = Evaluation(path_to_images=args.path_to_images,
                       model=detection_model,
                       model_name=model_name,
                       path_to_annotations=args.annotation,
                       batch_size=batch_size)

    click.echo(click.style(f"\n Start Evaluation \n", bold=True, fg='green'))    

    eval_mask.generate_detection_results_mask()
    
    click.echo(click.style(f"\n compute  bbox \n", bold=True, fg='green'))  
    eval_mask.COCO_process_mAP(type="bbox")

    click.echo(click.style(f"\n compute  segm \n", bold=True, fg='green'))

    eval_mask.COCO_process_mAP(type="segm")
       

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

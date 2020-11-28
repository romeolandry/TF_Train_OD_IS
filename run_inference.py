import argparse
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


from configs.run_config import *
from scripts.run.download import *
from scripts.run.make import *
from scripts.Evaluation.utils import *
from scripts.Evaluation.inference import Inefrence
from scripts.run.model  import Model


parser = argparse.ArgumentParser(description="Inference model. saved model or converted model")


parser.add_argument("--webcam", default=False, action="store_true",
    help=" Use web cam for inference")

parser.add_argument("--path_to_images", default=PATH_IMAGES +'/test2017' ,
    help="the path to directory of images or image path")

parser.add_argument("--number_of_image", default= 1 ,
    help="how much images in to directory schould inference")

parser.add_argument("-m","--path_to_model", required=True ,
    help="the path to model directory or  keras model")

parser.add_argument("--path_to_label", default=PATH_TO_LABELS_MAP ,
    help="the to label mab corresponding to model")

parser.add_argument("--checkpoint", default='ckpt-0',
    help= "run Inference from checkpoint. require if the web cam option is true" )


def main(args):
    model = Model(args.path_to_model,
                  args.checkpoint)
    if args.webcam:
        ## build model from checkpoint
        
        detection_model, model_name = model.build_detection_model()
        
        click.echo(click.style(f"\n Start inference using web cam\n", bg='green', bold=True, fg='white'))
        
        infer = Inefrence(path_to_images=args.path_to_images,
                          path_to_labels=args.path_to_label,
                          model=detection_model,
                          model_name=model_name)
        
        infer.inference_from_wedcam_with_checkpoint()      
    else:
        # Load model from saved model .pb

        detection_model, model_name = model.Load_model()
        click.echo(click.style(f"\n Start inferance from {args.path_to_images} ... \n", bg='green', bold=True, fg='white'))
        infer = Inefrence(path_to_images=args.path_to_images,
                          path_to_labels=args.path_to_label,
                          model=detection_model,
                          model_name=model_name)
        infer.infernce_images_from_dir(int(args.number_of_image))

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
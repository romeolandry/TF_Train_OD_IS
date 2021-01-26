import argparse
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


from configs.run_config import *
from scripts.run.download import *
from scripts.run.make import *
from scripts.Evaluation.utils import *
from scripts.Evaluation.inference import Inference
from scripts.run.model  import Model


parser = argparse.ArgumentParser(description="Inference model. saved model or converted model")

parser.add_argument("-m","--model", required=True ,
    help="the path to model directory(if image path to saved_model else path to model dir e.g ) or  keras model(.h5)")

parser.add_argument("-t",
                    "--type",
                    required=True,
                    choices= ['ssd', 'mask'],
                    help="choose between ssd and mask, which model you won to use")

parser.add_argument("-s","--size",
                    help=" model input size ")

parser.add_argument("--webcam", default=False, action="store_true",
    help=" Use web cam for inference")

parser.add_argument("-p","--path_to_images", default=PATH_IMAGES +'/test2017' ,
    help="the path to directory of images or image path")

parser.add_argument("-i","--nb_img", default= 1 ,
    help="how much images in to directory should inference")

parser.add_argument("-l","--label", default=PATH_TO_LABELS_MAP ,
    help="the to label mab corresponding to model")

parser.add_argument("-c","--checkpoint", default="ckpt-0",
    help="which checkpoint will be use e.g ckpt-0")

def main(args):
    model = Model(args.model,
                  args.checkpoint)
    if args.webcam:
        # build model from checkpoint        
        detection_model, model_name = model.build_detection_model()
        
        click.echo(click.style(f"\n Start inference using web cam\n", bold=True, fg='green'))
        
        infer = Inference(path_to_images=args.path_to_images,
                          path_to_labels=args.label,
                          model=detection_model,
                          model_name=model_name)
        
        if args.type == 'ssd':
            infer.ssd_inference_webcam()

        if args.type == 'mask':
            infer.mask_inference_webcam()

    else:
        # Load model from saved model .pb
        detection_model, model_name = model.Load_model()

        # image size for input model
        if not args.size:
            sys.stderr.write("specify the size input of your model")
        size = int(args.size)
        input_model_size = [size, size]

        click.echo(click.style(f"\n Start inference from {args.path_to_images} ... \n", bold=True, fg='green'))
        
        infer = Inference(path_to_images=args.path_to_images,
                          path_to_labels=args.label,
                          model=detection_model,
                          model_name=model_name,
                          model_image_size=input_model_size)       
        
        if args.type == 'ssd':
            infer.ssd_inference_image(int(args.nb_img))

        if args.type == 'mask':
            infer.mask_inference_image(int(args.nb_img))

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

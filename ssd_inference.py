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
from scripts.Evaluation.models  import Model
from scripts.Evaluation.ssd import Inference


parser = argparse.ArgumentParser(description="Inference model. saved model or converted model")

parser.add_argument("-m","--model", required=True ,
    help="the path to model directory(if image path to saved_model or freezed model")

parser.add_argument("-s","--size",
                    help=" model input size ")

parser.add_argument("--webcam", default=False, action="store_true",
    help=" Use web cam for inference")

parser.add_argument("--cam_input", default=camere_input,
    help="Index of availabe wedcam default 0")

parser.add_argument("--cam_width", default=camera_width,
    help="camera width")

parser.add_argument("--cam_height", default=camera_height,
    help="camera height")

parser.add_argument("-p","--path_to_images", default=PATH_IMAGES +'/test2017' ,
    help="the path to directory of images or image path")

parser.add_argument("-i","--nb_img", default= 1 ,
    help="how much images in to directory should inference")

parser.add_argument("-l","--label", default=PATH_TO_LABELS_MAP ,
    help="the to label mab corresponding to model")

parser.add_argument("--freezed", default=False, action="store_true",
    help=" Use freezed graph")

parser.add_argument("--th", default=.5 ,
    help=" Threshold for bounding box")

def main(args):
    # image size for input model
    if not args.size:
        sys.stderr.write("specify the size input of your model")
    
    size = int(args.size)
    input_model_size = [size, size]
    threshold = float(args.th)
    model = Model(args.model)

    if args.freezed:
        # Load model from forzen model .pb
        detection_model, model_name = model.load_freezed_model()
    else:
        # Load model from saved model .pb
        detection_model, model_name = model.Load_savedModel_model()


    infer = Inference(path_to_images=args.path_to_images,
                    path_to_labels=args.label,
                    model=detection_model,
                    model_name=model_name,
                    model_image_size=input_model_size,
                    threshold=threshold)
    if args.webcam:
        camera_input = args.cam_input
        camera_width = args.cam_input
        camera_height = args.cam_height

        click.echo(click.style(f"\n Start inference using webcam \n", bold=True, fg='green'))
        
        if args.freezed:        
            infer.ssd_inference_webcam_freezed_model(camera_input,
                                                     camera_width,
                                                     camera_height)
        else:
            sys.stderr.write("Opencv2 don't savedModel of Tensorflow. please use frozen graph")

    else:

        click.echo(click.style(f"\n Start inference from {args.path_to_images} ... \n", bold=True, fg='green'))    

        if args.freezed:
            infer.ssd_inference_freezed_model()
        else:
            infer.ssd_inference_image(int(args.nb_img))
       

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

import argparse
import click
import os

from configs.run_config import *
from scripts.run.download import *
from scripts.run.make import *
from scripts.Evaluation.utils import *
from scripts.Evaluation.models import *

# convert
parser = argparse.ArgumentParser(description="Convert SavedMode to inference or optimize the moedel with tf-trt")


parser.add_argument("-t","--type", choices=['tf_trt','freeze'],
    required=True,
    help=" convert to tf frozen graph(freeze) or tf-trt for inference")

parser.add_argument("-p","--path", required=True,
    help="path to savedModel to convert")

parser.add_argument("--mode", default= PRECISION_MODE,
    help="precision mode if you won a tf_trt model")

parser.add_argument("--max_ws", default=MAX_WORKSPACE_SIZE_BITES,
    help="MAX_WORKSPACE_SIZE_BITES for tf_trt model")

parser.add_argument("--input_size", default=640,
    help="Input size of image for eventual calibration")

parser.add_argument("--batch_size", default=32,
    help="batch-size to calibrate the data")


def main(args):
    if args.type =="freeze":
        click.echo(click.style(f"\n Conversion of {args.path} to Tensorflow inference model  \n", bold=True, fg='green'))
        # sys.stderr.write("Not available \n")
        con = Convertor(args.path)
        con.freeze_savedModel(image_size=int(args.input_size))
        
    else:
        click.echo(click.style(f"\n Conversion of {args.path} to Tensorflow-TensorRT model \n", bold=True, fg='green'))
        if ((args.mode == "INT8") or (args.mode == "int8")):
            click.echo(click.style(f"\n to calibrate you model the input image size should be the same as the input size for original model \t", bold=True, fg='white'))
            click.echo(click.style(f"\n to change the default image size set --input_size parameter \t", bold=True, fg='white'))
            click.echo(click.style(f"\n the calibration will be don with an image size of: {args.input_size} x {args.input_size} do you won to continuous ?", bold=True, fg='white'))
            
            c = click.getchar()
            click.echo()
            resp = str.capitalize(c)
            if resp=='N':
                raise("changed input size image")

        size = int(args.input_size)
        batched_input = batch_input(batch_size=args.batch_size, input_size=[size,size,3], path_to_test_img_dir='images/test2017')
        
        con = Convertor(args.path,args.mode,args.max_ws)
        original_path,saved_model_path = con.convert_to_TF_TRT_graph_and_save(calibration_data=batched_input)

        # save performance
        if args.type == 'tf_trt':
            mode = 'TensorFlow-TensorRT'
        else:
            mode = 'freeze'
        value = {
            'ORIGINAL': original_path,
            'PRECISION_MODE': args.mode,
            'MAX_WORKSPACE_SIZE_BITES': args.max_ws,
            'CONVERSION_TYPE': mode,
            'CONVERTED_MODEL':saved_model_path
        }
        model_name = os.path.basename(saved_model_path)
        to_save = {
            model_name:value,
        }
        save_performance('convertor',to_save)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
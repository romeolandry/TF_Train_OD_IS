from configs.run_config import *
from scripts.run.download import *
from scripts.run.make import *
from scripts.Evaluation.utils import *
from scripts.Evaluation.convertor import Convertor

import argparse

# convert
parser = argparse.ArgumentParser(description="Choose what you won beetwen [Data_preprocessing, Train, Evaluate]")


parser.add_argument("-t","--type", choices=['tftrt','trt'],
    required=True,
    help="would you convert to UFF_TRT or TF_TRTR")

parser.add_argument("--output_model_dir", default=PATH_TO_CONVERTED_MODELS,
    help="choose TF-TRT or TRT")

parser.add_argument("--precision_mode", default= PRECISION_MODE,
    help="precision mode for tensorrt model")

parser.add_argument("--max_ws", default=MAX_WORSPACE_SIZE_BITES,
    help="MAX_WORSPACE_SIZE_BITES for tenssorRT model")

parser.add_argument("-p","--path", required=True,
    help="path to model to convert .h5 or tensorflow model")

parser.add_argument("--input_size", default=640,
    help="Input size of image for eventual calibration")

parser.add_argument("--batch_size", default=32,
    help="batch-size to calibrate the data")


def main(args):
    if args.type =="trt":
        click.echo(click.style(f"\n Conversion of {args.path} to TensorRT using Uff parser model \n", bg='green', bold=True, fg='white'))
    else:
        click.echo(click.style(f"\n Conversion of {args.path} to Tensorflow-TensorRT model \n", bg='green', bold=True, fg='white'))
        if ((args.precision_mode == "INT8") or (args.precision_mode == "int8")):
            click.echo(click.style(f"\n to calibrate you model the input image size should be the same as the input size for original model \t", bold=True, fg='white'))
            click.echo(click.style(f"\n to change the default image size set --input_size parameter \t", bold=True, fg='white'))
            click.echo(click.style(f"\n the calibration will be don wiht an image size of: {args.input_size} x {args.input_size} do you won to continuous ?", bold=True, fg='white'))
            
            c = click.getchar()
            click.echo()
            resp = str.capitalize(c)
            if resp=='N':
                raise("changed input size image")


        batched_input = batch_input(batch_size=args.batch_size, input_size=[args.input_size,args.input_size,3], path_to_test_img_dir='images/test2017')

        con = Convertor(args.path,args.precision_mode,args.max_ws)
        con.convert_to_TF_TRT_graph_and_save(calibration_data=batched_input)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
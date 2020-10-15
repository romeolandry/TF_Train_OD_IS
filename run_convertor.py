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

parser.add_argument("-p","--path", required=True, help="path to model to convert .h5 or tensorflow model")


def main(args):
    if args.type =="trt":
        click.echo(click.style(f"\n Conversion of {args.path} to TensorRT using Uff parser model \n", bg='green', bold=True, fg='white'))
    else:
        click.echo(click.style(f"\n Conversion of {args.path} to Tensorflow-TensorRT model \n", bg='green', bold=True, fg='white'))
        batched_input = batch_input(batch_size=32, input_size=[640,640,3], path_to_test_img_dir='images/test2017')
        con = Convertor(args.path,args.precision_mode,args.max_ws)
        con.convert_to_TF_TRT_graph_and_save(calibration_data=batched_input)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
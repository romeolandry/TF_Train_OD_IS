import argparse
import click
import os
import tensorflow as tf


from configs.run_config import *
from scripts.Evaluation.utils import *
from scripts.Evaluation.models import *

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

# convert
parser = argparse.ArgumentParser(description="Convert SavedMode to inference or optimize the moedel with tf-trt")


parser.add_argument("-t","--type", choices=['tf_trt','freeze'],
    required=True,
    help=" convert to tf frozen graph(freeze) or tf-trt for inference")

parser.add_argument("--mask", default=False, action="store_true",
    help=" to freeze mask")

parser.add_argument("-m","--model",type=str,
    required=True,
    help="path to savedModel to convert")

parser.add_argument("--mode", type=str,
    default= PRECISION_MODE, choices= ACCEPTED_MODE,
    help="precision mode for tf_trt model")

parser.add_argument("--max_ws", type=int,
    default=MAX_WORKSPACE_SIZE_BITES,
    help="MAX_WORKSPACE_SIZE_BITES for tf_trt model")

parser.add_argument("--min_seg_size", type=int,
    default=MIN_SEGMENTATION_SIZE,
    help="Min Segmentation size for tf_trt model")

parser.add_argument("--input_size", type=int,
    default=640,
    help="Input size of image for eventual calibration")

parser.add_argument("--input_data_dir", type=str,
    default=PATH_IMAGES +'/val2017',
    help=" Vaildation Input directory  of image for eventual calibration")

parser.add_argument("--input_type", type=str,
    default=INPUT_TYPE_MODEL,
    help=" type of input tensor (int or float)")

parser.add_argument("--annotation_file", type=str,
    default=PATH_ANNOTATIONS +'/instances_val2017.json',
    help=" path to coco annotation file to load  input file")

parser.add_argument("--batch_size",type=int, 
    default=1,
    help="batch-size to calibrate the data")

parser.add_argument("--build_eng",
    default=False, action="store_true",
    help=" Use freezed graph")

def main(args):
    con = Convertor(path_to_model=args.model,
                    max_workspace_size_bytes=args.max_ws,
                    min_seg_size= args.min_seg_size,
                    precision_mode=args.mode,
                    input_size=args.input_size,
                    val_data_dir=args.input_data_dir,
                    annotation_file=args.annotation_file,
                    batch_size= args.batch_size,
                    model_input_type= args.input_type,
                    build_engine = args.build_eng)
    if args.type =="freeze":
        click.echo(click.style(f"\n Conversion of {args.path} to Tensorflow inference model  \n", bold=True, fg='green'))
        # sys.stderr.write("Not available \n")
        if args.mask:
            con.freeze_savedModel_mask()
        else:
            con.freeze_savedModel_ssd()
        
    else:
        click.echo(click.style(f"\n Conversion of {args.model} to Tensorflow-TensorRT model \n", bold=True, fg='green'))
        if ((args.mode == "INT8") or (args.mode == "int8")):
            click.echo(click.style(f"\n to calibrate you model the input image size should be the same as the input size for original model \t", bold=True, fg='white'))
            click.echo(click.style(f"\n to change the default image size set --input_size parameter \t", bold=True, fg='white'))
            click.echo(click.style(f"\n the calibration will be don with an image size of: {args.input_size} x {args.input_size} do you won to continuous(Y or N) ?", bold=True, fg='white'))
            
            c = click.getchar()
            click.echo()
            resp = str.capitalize(c)
            if resp=='N':
                raise("changed input size image")
        

        model_name,saved_model_path, time = con.convert_to_TF_TRT_graph_and_save()

        # save performance
        if args.type == 'tf_trt':
            mode = 'TensorFlow-TensorRT'
        else:
            mode = 'freeze'
        value = {
            'CONVERTED_MODEL': model_name,
            'PRECISION_MODE': args.mode,
            'MAX_WORKSPACE_SIZE_BITES': args.max_ws,
            'MIN_SEGMENTATION_SIZE': args.min_seg_size,
            'CONVERSION_TYPE': mode,
            'CONVERTED_MODEL_location':saved_model_path,
            'CONVERSION_TIME': time
        }
        model_name = os.path.basename(saved_model_path)
        to_save = {
            model_name:value,
        }
        save_performance('convertor',to_save)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
from configs.run_config import *
import argparse
from scripts.run.download import *
from scripts.run.make import *
import click

"""
"""

parser = argparse.ArgumentParser(description="Choose what you won beetwen [Data_preprocessing, Train, Evaluate]")

parser.add_argument("--data_preprocessing", default=False, action="store_true",
    help="If you already have coco Dataset downloaded and just won to create\
    the TF-record of your data.")

parser.add_argument("--mask", default=False, action="store_true",
    help="Record for mask else bbox .")

parser.add_argument("-a","--action",choices=['train','eval','export'],
    help="Choose which action to percist training evaluation or export")

parser.add_argument("-m","--model",choices=list(set(LIST_MODEL_TO_DOWNLOAD.keys())),
    help="Choose which Model you wont to train")

## Evaluation and export

parser.add_argument("--model_dir",
    help="Path to output model directory where event will be written or exported.")

parser.add_argument("--pipeline_config",
    help="Path to pipeline config file.")

parser.add_argument("--checkpoint_dir",
    help="Path to directory holding a checkpoint.")

parser.add_argument("--eval_timeout", default=1,
    help="Path to directory holding a checkpoint.")




def main(args):

    print(args)

if __name__ == "__main__":
    args = parser.parse_args()
    if(args.data_preprocessing):
        click.echo(click.style(f"\n COCO DataSet 2017 will be preprocessed \n", bold=True, fg='green'))
        if not (os.path.exists(cfg.PATH_ANNOTATIONS) or os.path.exists(cfg.PATH_IMAGES)):
            click.echo('The directory image/annotations doesn\'t exist. If you still have downloaded images/ and annotations please type n to skip and make sure you configure directories correctly: [yn] ', nl=False,)
            c = click.getchar()
            click.echo()
            resp = str.capitalize(c)
            if resp=='Y':
                # Download Coco Dataset
                download_coco()
            else:
                raise("set directory for data")
        # Do preprocessing
        click.echo(click.style(f"\n Create of tf record \n", bold=True, fg='blue'))
        if args.mask:
            success = make_preprocessing("mask")
        else:
            success = make_preprocessing("bbox")
        if(success):
            click.echo(click.style(f"\n tf record created and saved in to {PATH_ANNOTATIONS} directory \n", bold=True, fg='blue'))
    
    if (args.action == 'train'):
        if(args.model):
            assert args.model,f"You should select one models beetwen {LIST_MODEL_TO_DOWNLOAD.keys()}"
            success = download_pre_trained_model(args.model)
            if success == -1:
                click.echo(click.style(f"\n the model {args.model} couldn't be downloaded. Please verify that the url is still valid \n", bg='red', bold=True, fg='white'))
                exit()
            if success == 0:
                click.echo(click.style(f"\n Set configuration in to pipeline.config \n", bold=True, fg='red'))
                exit()
            
            # Train Model
            click.echo(click.style(f"\n Proceed of Train of {args.model} \n", bold=True, fg='green'))
            train = make_train(args.model)
    else:
        assert args.model_dir,"Path to output model directory where event will be written or exported."
        assert args.pipeline_config,"Path to pipeline config file."
        assert args.checkpoint_dir,"Path to directory holding a checkpoint."
        # Evaluate
        if(args.action == 'eval'):
            click.echo(click.style(f"\n Proceed of Evaluation  of {args.model} \n", bold=True, fg='green'))
            make_eval(args.model_dir,
                      args.pipeline_config,
                      args.checkpoint_dir,
                      args.eval_timeout)
            
        # Export trainned Model
        if (args.action == 'export'):
            click.echo(click.style(f"\n Export  {args.model} \n", bold=True, fg='green'))
            exported = make_export(args.model_dir,
                                   args.pipeline_config,
                                   args.checkpoint_dir)
            
            
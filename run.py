from configs.run_config import *
import argparse
from scripts.run.download import *
from scripts.run.make import *
import click

"""
"""

parser = argparse.ArgumentParser(description="Choose what you won beetwen [Data_preprocessing, Train, Evaluate]")

parser.add_argument("--data_preprocessing", default=False, action="store_true",
    help="If you already have coco Dataset downloaded and just to create\
    the TF-record of your data.")    

parser.add_argument("-m","--model",choices=list(set(LIST_MODEL_TO_DOWNLOAD.keys())),
    help="Choose which Model you wont to train")

parser.add_argument("--eval",choices=list(set(LIST_MODEL_TO_DOWNLOAD.keys())),
    help="Choose which Model you wont to Evaluate. the model that is in train can also been choosed.")



def main(args):

    print(args)

if __name__ == "__main__":
    args = parser.parse_args()
    if(args.data_preprocessing):
        click.echo(click.style(f"\n COCO DataSet 2017 will be preprocessed \n", bg='green', bold=True, fg='white'))
        if not (os.path.exists(cfg.PATH_ANNOTATIONS) or os.path.exists(cfg.PATH_IMAGES)):
            click.echo('The directory image/annotations doesn\'t exist. If you still have downloaded images/ and annotations please type n to skip and make shure you configure directories correctly: [yn] ', nl=False,)
            c = click.getchar()
            click.echo()
            resp = str.capitalize(c)
            if resp=='Y':
                # Donwload Coco Dataset
                download_coco()
            else:
                raise("set directoy for data")
        # Do preprocessing
        click.echo(click.style(f"\n Create of tf record \n", bg='blue', bold=True, fg='white'))
        success = make_preprocessing()
        if(success):
            click.echo(click.style(f"\n tf record created and saved in to {PATH_ANNOTATIONS} directory \n", bg='blue', bold=True, fg='white'))
    else:
        ## run evalution of the selected model
        if(args.eval):
            assert args.eval,f"You should select one models beetwen {LIST_MODEL_TO_DOWNLOAD.keys()}"
            make_eval_on_train(args.eval)
        else:
            assert args.model,f"You should select one models beetwen {LIST_MODEL_TO_DOWNLOAD.keys()}"
            success = download_pre_trained_model(args.model)
            if success == -1:
                click.echo(click.style(f"\n the model {args.model} coudn\'t be downloaded. Please verify that the url is still valid \n", bg='red', bold=True, fg='white'))
                exit()
            if success == 0:
                click.echo(click.style(f"\n Set configuration in to pipeline.config \n", bg='red', bold=True, fg='white'))
                exit()
        
            ## Train Model
            click.echo(click.style(f"\n Proceed of Train of {args.model} \n", bg='green', bold=True, fg='white'))
            train = make_train(args.model)
            if train :
                click.echo(click.style(f"\n Export  {args.model} \n", bg='green', bold=True, fg='white'))
                ## do export
                exported = make_export(args.model)
            
            
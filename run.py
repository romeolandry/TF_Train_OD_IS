from configs.run_config import *
import argparse
from scripts.run.download import *
import click

"""
"""

parser = argparse.ArgumentParser(description="Choose what you won beetwen [Data_preprocessing, Train, Evaluate]")

parser.add_argument("--data_preprocessing", default=False, action="store_true",
    help="If you already have coco Dataset downloaded and just to create\
    the TF-record of your data.")
## Coco data preprocessing

parser.add_argument("--img_dir", default=PATH_IMAGES,
    help="directory should content subdirectory [test, train, val].")

parser.add_argument("--annot_dir", default=PATH_ANNOTATIONS,
    help="should content annotation file for [test, train, val]")

parser.add_argument("--tf_record",default=PATH_ANNOTATIONS,
    help=".record file create during the preprocesing of data")

# Train or evaluate
parser.add_argument("--eval", default=False, action="store_true",
    help="--eval if you wont to evaluate some model. by default the choosen Model will be Train")
parser.add_argument("-m","--model",choices=list(set(LIST_MODEL_TO_DOWNLOAD)),
    help="Choose which Model you wont to train or Evaluate")


def main(args):

    print(args)

if __name__ == "__main__":
    args = parser.parse_args()
    if(args.data_preprocessing):
        click.echo(click.style(f"\n COCO DataSet 2017 will be preprocessed \n", bg='green', bold=True, fg='white'))
        if(os.path.exists(cfg.PATH_ANNOTATIONS) or os.path.exists(cfg.PATH_IMAGES)):
            ## Do Pre processing
            click.echo(click.style(f"\n Create of tf record \n", bg='blue', bold=True, fg='white'))
        else:
            
            click.echo('The directory image/annotations doesn\'t exist. If you still have downloaded images/ and annotations please\
                    type n to skip and make shure you configure directories correctly: [yn] ', nl=False,)
            c = click.getchar()
            click.echo()
            resp = str.capitalize(c)
            if resp=='N':
                exit("already downloaded")
            else:
                # Donwload Coco Dataset
                download_coco_2017(2017)
                # Do preprocessing
                      

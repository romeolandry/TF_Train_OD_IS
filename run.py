from configs.run_config import *
import argparse

"""
"""

parser = argparse.ArgumentParser(description="Choose what you won beetwen [Data_preprocessing, Train, Evaluate]")

## Coco data preprocessing
parser.add_argument("--download-data", default=False, action="store_true",
    help="set if wont to download COCO DataSet, this download data and run the preprocessing process")

parser.add_argument("--data-preprocessing", default=False, action="store_true",
    help="If you already have coco Dataset downloaded and just to create\
    the TF-record of your data.")

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
    
    main(LIST_MODEL_TO_DOWNLOAD)
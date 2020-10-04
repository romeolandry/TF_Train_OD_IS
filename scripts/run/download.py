import os
import sys
#import requests

import tarfile
import urllib.request

sys.path.append(os.path.abspath(os.curdir))

import configs.run_config as cfg

def pre_trained_model (model_name):

    file_name = (model_name.split("/")[-1]).split(".")[0] # get file name from url
    # create directory if not exit
    dir_path = os.path.join(cfg.PRE_TRAINED_MODEL_DIR_PATH,file_name)
    if not os.path.exists(dir_path):
        print("___________________ Downloading the model_____________")
        ftpstream = urllib.request.urlopen(model_name)
        content = tarfile.open(fileobj=ftpstream, mode="r|gz")
        content.extractall(cfg.PRE_TRAINED_MODEL_DIR_PATH)
        print(f"The model was correctly downloaded and saved in to {dir_path}")
    else:
        print("This Model was alredy downloaded!")

import os
import sys
import configparser

import numpy as np
import tensorflow as tf
import glob
from tensorflow.keras.preprocessing import image
from PIL import Image

from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions

sys.path.append(os.path.abspath(os.curdir))

from configs.run_config import *
from scripts.api_scrpit import *


def load_img_from_folder(path_folder):
    img_list = []
    for filename in glob.glob(path_folder + '/*.jpg'):
        img = Image.open(filename)
        img_list.append(img)
    return img_list

def predict_and_benchmark_throughput(batched_input, infer, N_warmup_run=50, N_run=1000):
    elapsed_time = []
    all_preds = []
    batch_size = batched_input.shape[0]

    for i in range(N_warmup_run):
        labeling = infer(batched_input)
        preds = labeling['predictions'].numpy()

    for i in range(N_run):
        start_time = time.time()

        labeling = infer(batched_input)

        preds = labeling['predictions'].numpy()

        end_time = time.time()

        elapsed_time = np.append(elapsed_time, end_time - start_time)
        
        all_preds.append(preds)

        if i % 50 == 0:

            print('Steps {}-{} average: {:4.1f}ms'.format(i, i+50, (elapsed_time[-50:].mean()) * 1000))

    print('Throughput: {:.0f} images/s'.format(N_run * batch_size / elapsed_time.sum()))
    totall_time = N_run * batch_size / elapsed_time.sum()
    return all_preds, totall_time

def batch_input (batch_size=8, input_size=[299,299,3], path_to_test_img_dir=''):

    #list_img =  load_img_from_folder(path_to_test_img_dir)

    batched_input = np.zeros((batch_size,input_size[0],input_size[1],input_size[2]), dtype=np.float32)
   
    for i in range(1,batch_size+1):
        index = str('%d' % (i % 1000)).zfill(12) # 581918
        img_name = index + '.jpg'
        img_name= format(img_name)
        img_path =  path_to_test_img_dir + '/' + img_name
        img = image.load_img(img_path, target_size=(input_size[0],input_size[1]))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        batched_input[i, :] =x
        batched_input = tf.constant(batched_input) 
        return batched_input

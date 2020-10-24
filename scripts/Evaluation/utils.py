import os
import sys
import json

import numpy as np
import tensorflow as tf
import glob
from tensorflow.keras.preprocessing import image
from PIL import Image

from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions

sys.path.append(os.path.abspath(os.curdir))

from configs.run_config import *
from scripts.api_scrpit import *


def load_img_from_folder(path_folder, anzahl = None, mAP = False, bacth_size= 32,image_size = [640,640]):
    """
        load image from file into numpy array.

        put each image into an numpy array  to feed tensorflow.
        the numpy shape have the following struckture.
        (height, width, channels) channels=3

        Args:
        path to image folder
        number of image to load. default is None all the image will been loaded

        Returns:
           list of  uint8 numpy array with shape (img_height, img_width, 3)        
    """
    img_list = []
    batch_number = 0
    count = 0
    total_file = len(glob.glob1(path_folder + '/','*.jpg'))
    total_loaded = 0
    if anzahl is not None:
        total_file = anzahl
    
    if os.path.isdir(path_folder):
        for filename in glob.glob(path_folder + '/*.jpg'):
            img = Image.open(filename).resize((image_size[0],image_size[1]))
            if not mAP:
                img_list.append(np.array(img))
            else:
                # get Id of image
                imageId = int(filename.split('.')[0].split('/')[-1]) 
                img_list.append({'imageId':imageId,
                                 'np_image':np.array(img)})
            count +=1
            if (count == total_file):
                yield img_list
            
            if (count % bacth_size == 0):
                yield img_list
    else:
        img = Image.open(path_folder)
        img_list.append(np.array(img))
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

def save_perfromance(status_to_save, json_daten,file_name=None):
    if not os.path.isdir('performances'):
        os.mkdir('performances')
    
    if status_to_save == "convertor":

        if not os.path.isfile(PATH_PERFORMANCE_CONVERT):
            with open (PATH_PERFORMANCE_CONVERT,'w+') as json_file:
                json.dump(json_daten,json_file)
        else:
            with open (PATH_PERFORMANCE_CONVERT,'r+') as js_file:
                data = json.load(js_file)
                data.update(json_daten)
                js_file.seek(0)
                json.dump(data,js_file)

    if(status_to_save=='prediction'):
        with open (os.path.join(PATH_PERFORMANCE_INFER,file_name),'w+') as json_file:
            json_file.write(json.dumps(json_daten, indent=4))
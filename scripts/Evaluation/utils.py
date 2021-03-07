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
import cv2

from pycocotools.coco import COCO 

#from scripts.api_scrpit import *

"""
    Create tensorflow dataset for coco validation data.

    put each image tf.data.Dataset  to feed tensorflow.
    
    (height, width, channels) channels=3

    Args:
        path to one image or a folder conting images
        number of image to load. default is None all the image will been loaded

    Returns:
        tf.data.Dataset and list of image ids       
"""

def load_img_from_folder_update(path_folder,
                                annotations_path,
                                batch_size,
                                input_size,
                                dtype=tf.uint8):

    coco = COCO(annotation_file=annotations_path)
    image_ids = coco.getImgIds() 
    image_paths = []
    for image_id in image_ids:
        coco_img=coco.imgs[image_id]
        image_paths.append(os.path.join(path_folder, coco_img['file_name']))
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)

    def preprocess_fn(path):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        if input_size is not None:
            image = tf.image.resize(image, size=(input_size,input_size))
            image = tf.cast(image, dtype)
        return image
        print(f" image size {image.shape}")
    dataset = dataset.map(map_func=preprocess_fn, num_parallel_calls=8)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(count=1)
    
    return dataset , image_ids




"""
    load image from file into numpy array.

    put each image into an numpy array  to feed tensorflow.
    the numpy shape have the following structure.
    (height, width, channels) channels=3

    Args:
        path to one image or a folder conting images
        number of image to load. default is None all the image will been loaded

    Returns:
        list of  uint8 numpy array with shape (img_height, img_width, 3)        
"""

def load_img_from_folder(path_folder,
                        validation_split=0.1,
                        mAP = False,
                        batch_size= 32):
    
    img_list = []
    batch_number = 0
    count = 0
    total_file = len(glob.glob1(path_folder + '/','*.jpg'))

    total_file =  total_file * validation_split
    
    ndx = 0
    if not os.path.isdir(path_folder):
        sys.stderr.write("Image folder is not a directory")

    for filename in glob.glob(path_folder + '/*.jpg'):
        img = Image.open(filename).resize((image_size[0],image_size[1]))
        if(count > total_file):
            break

        if not mAP:
            img_list.append(np.array(img))
        else:
            # get Id of image
            imageId = int(filename.split('.')[0].split('/')[-1]) 
            img_list.append({'imageId':imageId,
                                'np_image':np.array(img)})
        
        if (len(img_list) and (len(img_list) % batch_size) == 0):
            yield img_list[ndx:min(ndx + batch_size,len(img_list))]
            ndx = ndx + batch_size

        count = count + 1


    """
        load image from file into numpy array.

        put each image into an numpy array  to feed tensorflow.
        the numpy shape have the following structure.
        (height, width, channels) channels=3

        Args:
        path to image folder
        number of image to load. default is None all the image will been loaded

        Returns:
           list of  uint8 numpy array with shape (img_height, img_width, 3)        
    """

def load_img_from_folder_for_infer(path_folder,
                                   number_of_images = None,
                                   image_size = [640,640]):
    img_list = []
    count = 0

    if os.path.isdir(path_folder):
        for filename in glob.glob(path_folder + '/*.jpg'):
            img = Image.open(filename)
            img_list.append(np.array(img))

            count +=1
            if (number_of_images is not None) and (count == number_of_images):
                break
        return img_list
    else:
        img = Image.open(path_folder)
        img_list.append(np.array(img))
        return img_list


''' def predict_and_benchmark_throughput(batched_input, infer, N_warmup_run=50, N_run=1000):
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
    return all_preds, totall_time '''

def batch_input (batch_size=8, input_size=[299,299,3], path_to_test_img_dir=''):

    batched_input = np.zeros((batch_size,input_size[0],input_size[1],input_size[2]), dtype=np.float32)
   
    for i in range(1,batch_size+1):
        index = str('%d' % (i % 1000)).zfill(12) # 000000000001
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

def save_performance(status_to_save, json_data,file_name=None):
    if not os.path.isdir('performances'):
        os.mkdir('performances')
    
    if status_to_save == "convertor":

        if not os.path.isfile(PATH_PERFORMANCE_CONVERT):
            with open (PATH_PERFORMANCE_CONVERT,'w+') as json_file:
                json.dump(json_data,json_file)
        else:
            with open (PATH_PERFORMANCE_CONVERT,'r+') as js_file:
                data = json.load(js_file)
                data.update(json_data)
                js_file.seek(0)
                json.dump(data,js_file)

    if(status_to_save=='prediction'):
       
        with open (os.path.join(PATH_PERFORMANCE_INFER,file_name),'w+') as json_file:
            json_file.write(json.dumps(json_data, indent=4))

''' 
    Read label text and it as diction key value
'''
def read_label_txt (path):
    count = 0
    categories= {}

    with open(path, "r") as f:
        Lines = f.readlines()
                
        for line in Lines:
            count = count + 1
            categories.update({count:line.strip()})
    return categories

def set_input_camera(camera_input,camera_width,camera_height):
    cap = cv2.VideoCapture(camera_input)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1080)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,700)
    return cap
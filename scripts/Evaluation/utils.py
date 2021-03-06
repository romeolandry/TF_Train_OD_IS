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
                        batch_size= 32,
                        input_size=None):
    
    img_list = []
    batch_number = 0
    count = 0
    total_file = len(glob.glob1(path_folder + '/','*.jpg'))

    total_file =  total_file * validation_split
    
    ndx = 0
    if not os.path.isdir(path_folder):
        sys.stderr.write("Image folder is not a directory")

    for filename in glob.glob(path_folder + '/*.jpg'):
        img = Image.open(filename)
        if input_size is not None:
            img = img.resize((input_size,input_size))
        if(count > total_file):
            break

        if not mAP:
            img_list.append(np.array(img))
        else:
            # get Id of image
            imageId = int(filename.split('.')[0].split('/')[-1]) 
            img_list.append({'imageId':imageId,
                             'np_image':np.array(img),
                             'file_path':filename})
        
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
def read_label_txt(path):
    count = 0
    categories= {}

    with open(path, "r") as f:
        Lines = f.readlines()
                
        for line in Lines:
            count = count + 1
            categories.update({count:line.strip()})
    return categories

def set_input_camera(camera_input,camera_width,camera_height,file_name):
    cap = cv2.VideoCapture(camera_input)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1080)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,700)
    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.17
    out_file = cv2.VideoWriter(file_name,cv2.VideoWriter_fourcc('M','J','P','G'), 10, (int(cap.get(3)),int(cap.get(4))))
    return cap, out_file


def draw_iou(image, ground_truth, pred_bbox, iou):
    cv2.rectangle(image, (int(ground_truth[0]), int(ground_truth[1])), (int(ground_truth[2]), int(ground_truth[3])), (125, 255, 51), thickness=2)
    cv2.putText(image, " ground truth",(int(ground_truth[0])+10,int(ground_truth[1])-5),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (125, 255, 51), 2)
                        
    cv2.rectangle(image, (int(pred_bbox[0]), int(pred_bbox[1])), (int(pred_bbox[2]), int(pred_bbox[3])), (0, 0, 255), thickness=2)
    cv2.putText(image, " predicted IUO = " + str(iou) ,(int(pred_bbox[2]) -10,int(pred_bbox[3])+ 20),cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 0, 255), 1)

    return image

def parse_detector(image,boxes,classes,scores,categories,list_object_to_tracked,masks=None,tp_th=.50):
    extracted = []
    if masks is None:
        assert boxes.shape[0]  == classes.shape[0] == scores.shape[0]
    else:
        assert boxes.shape[0]  == classes.shape[0] == scores.shape[0] == masks.shape[0]

    h,w= image.shape[:2]
    for i in range(boxes.shape[0]):
        classId = classes[i]
        score = scores[i]
        # get class text
        label = categories[classId]
        if label not in list_object_to_tracked:
            continue
        box = boxes[i]* np.array([w,h,w,h])
        if score> tp_th:
            if masks is None:
                extracted.append({
                    "class": label,
                    "score": str(score),
                    "box":list(box)
                })
            else:
                mask = mask[i]

                extracted.append({
                    "class": label,
                    "score": str(score),
                    "box":list(box),
                    "segmentation":mask
                })
                

    return extracted
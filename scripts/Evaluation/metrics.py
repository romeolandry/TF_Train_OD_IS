import os
import sys
import time
import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import tensorflow as tf
import numpy as np
sys.path.append(os.path.abspath(os.curdir))

from configs.run_config import *
from scripts.Evaluation.utils import *
from scripts.run.model  import Model
from object_detection.utils import ops as utils_ops

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

class Evaluation:
    def __init__(self,
                 path_to_images,
                 model,
                 model_name,
                 path_to_annotations,
                 batch_size=32,):
        self.__path_to_images = path_to_images
        self.__model = model
        self.__model_name = model_name
        self.__path_to_annotations = path_to_annotations
        self.__batch_size = batch_size


    def predict_and_benchmark_throughput(self,batched_input_image):
        elapsed_time = []
        all_detections = []
        batch_size = batched_input_image.shape[0]

        # warm up
        for i in range (self.__N_warm_up_run):
            detection = self.__model(batched_input_image)
            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
                          for key, value in detections.items()}

            detections['num_detections'] = num_detections
            # detection_classes should be int64.
            #detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        for i in range(self.__N_run):
            start_time = time.time()

            detections = self.__model(batched_input_image)
            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
                          for key, value in detections.items()}

            detections['num_detections'] = num_detections

            end_time = time.time()

            elapsed_time = np.append(elapsed_time, end_time - start_time)

            all_detections.append(detections)

            if i % self.__N_warm_up_run == 0:
                print('Steps {}-{} average: {:4.1f}ms'.format(i, i+self.__N_warm_up_run, (elapsed_time[-self.__N_warm_up_run:].mean()) * self.__N_run))
            
        print('Throughput: {:.0f} images/s'.format(N_run * batch_size / elapsed_time.sum()))
        totall_time = N_run * batch_size / elapsed_time.sum()
        return all_detections, totall_time

    """
        Run detection on each image an write result into Json file
    """
    def generate_detection_results_mask(self):
        elapsed_time = []
        results = []
        total_image = 0
        batch_count = 0

        for images in load_img_from_folder(self.__path_to_images,number_of_images=5, batch_size=self.__batch_size, mAP=True):
            # convert images to be a tensor
            batch_count = batch_count + 1
            print(f"run evaluation for batch {batch_count} of {len(images)} images \t")
            for item in images:
                # convert images to be a tensor
                input_tensort = tf.convert_to_tensor(item['np_image'])
                input_tensort = input_tensort[tf.newaxis, ...]
                input_tensort = tf.convert_to_tensor(np.expand_dims(item['np_image'], 0), dtype=tf.uint8)
            
                label_id_offset = 1

                start_time = time.time()
                detections = self.__model(input_tensort)
                end_time = time.time()

                if 'detection_masks' in detections:
                    detection_masks = tf.convert_to_tensor(detections['detection_masks'][0])
                    detection_boxes = tf.convert_to_tensor(detections['detection_boxes'][0])

                    # Reframe the the bbox mask to the image size.

                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks, detection_boxes,item['np_image'].shape[0], item['np_image'].shape[1])
                    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,tf.uint8)
                    detections['detection_masks_reframed'] = detection_masks_reframed.numpy()

                elapsed_time = np.append(elapsed_time, end_time - start_time)              

                for boxes, classes, score in zip (detections['detection_boxes'][0].numpy(),
                                                  (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
                                                  detections['detection_scores'][0].numpy()):
                    height, width,_ = item['np_image'].shape
                    x = float(boxes[0]*height)
                    y = float(boxes[1]*width)
                    w = float(boxes[2]*height)
                    h = float(boxes[3]*width)
                    results.append({'image_id':item['imageId'],
                                    'category_id': int(classes),
                                    'bbox':[x, y, w, h],
                                    'score': float(score)})
                total_image = len(images)
            print('average time pro batch: {:4.1f} ms'.format((elapsed_time[-len(images):]/self.__batch_size) * 1000 ))

            ## save predicted annotation
            print(f"Total evaluate {len(images)} \t")
            print(f"save results in to json!")
            save_performance('prediction', json_data=results, file_name= self.__model_name +'.json')
            results.clear()
        print(f'total time {elapsed_time.sum()}')
        print('After all Evaluation FPS {:4.1f} ms '.format(1000/(sum(elapsed_time)/len(elapsed_time)*1000)))
    
    """
        Run detection on each image an write result into Json file
    """
    def generate_detection_results_ssd(self):
        elapsed_time = []
        results = []
        total_image = 0
        batch_count = 0

        for images in load_img_from_folder(self.__path_to_images,number_of_images=None, batch_size=self.__batch_size, mAP=True):
            # convert images to be a tensor
            batch_count = batch_count + 1
            print(f"run evaluation for batch {batch_count} of {len(images)} images \t")
            for item in images:
                input_tensort = tf.convert_to_tensor(item['np_image'])
                input_tensort = input_tensort[tf.newaxis,...]
                try:
                    start_time = time.time()

                    detections = self.__model(input_tensort)
                    num_detections = int(detections.pop('num_detections'))
                    detections = {key: value[0, :num_detections].numpy()
                                for key, value in detections.items()}
                    detections['num_detections'] = num_detections
                    # convert detection  classes to int
                    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
                    end_time = time.time()

                    elapsed_time = np.append(elapsed_time, end_time - start_time)
                except :
                    continue                

                for boxes, classes, score in zip (detections['detection_boxes'],
                                                detections['detection_classes'],
                                                detections['detection_scores']):
                    height, width,_ = item['np_image'].shape
                    x = float(boxes[0]*height)
                    y = float(boxes[1]*width)
                    w = float(boxes[2]*height)
                    h = float(boxes[3]*width)
                    results.append({'image_id':item['imageId'],
                                    'category_id': int(classes),
                                    'bbox':[float(boxes[0]), float(boxes[1]), float(boxes[2]), float(boxes[3])],
                                    'score': float(score)})
                total_image = len(images)
                print('average time pro batch: {:4.1f} ms'.format((elapsed_time[-len(images):].mean()) * 1000 ))

            ## save predicted annotation 
            print(f"Total evaluate {len(images)} \t")
            print(f"save results in to json!")
            save_performance('prediction', json_data=results, file_name= self.__model_name +'.json')
            results.clear()

        print(f'total time {elapsed_time.sum()}')
        print('After all Evaluation FPS {:4.1f} ms '.format(1000/(sum(elapsed_time)/len(elapsed_time)*1000)))

    def COCO_process_mAP(self,type):
        cocoGt = COCO(self.__path_to_annotations)
        cocoDt = cocoGt.loadRes(os.path.join(PATH_PERFORMANCE_INFER,self.__model_name + '.json'))
        
        imgIds = sorted(cocoGt.getImgIds())

        cocoEval = COCOeval(cocoGt, cocoDt, type)
        
        cocoEval.params.imgIds = imgIds

        cocoEval.evaluate()
        cocoEval.accumulate()

        print(cocoEval.summarize())



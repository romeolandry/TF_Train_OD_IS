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


"""
    Post preprocessing ssd
"""
def postprocessing(img, boxes,classes, scores, th=0.1):
    h, w, _ = img.shape
   
    output_boxes = boxes * np.array([h,w,h,w])
    output_boxes = output_boxes.astype(np.int32)
    output_boxes = output_boxes[:,[1,0,3,2]] # wrap x's y's each box[x0,y0,x1,y1]=> box[y0,x0,y1,x1]
    output_confs = scores
    output_cls =classes.astype(np.int32)

    # return bboxes with confidence score above threshold
    mask = np.where(output_confs >= th)
    return output_boxes[mask], output_cls[mask], output_confs[mask]

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

    """
        Run detection on each image an write result into Json file
    """
    def generate_detection_results_mask(self):
        elapsed_time = []
        results = []
        total_image = 0
        batch_count = 0

        for images in load_img_from_folder(self.__path_to_images,validation_split=1, batch_size=self.__batch_size, mAP=True):
            # convert images to be a tensor
            batch_count = batch_count + 1
            print(f"run evaluation for batch {batch_count} \t")
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
                    detection_masks_reframed = tf.cast(detection_masks_reframed > 1e-2,tf.uint8)
                    detections['detection_masks_reframed'] = detection_masks_reframed.numpy()

                elapsed_time = np.append(elapsed_time, end_time - start_time)

                print(f" maskered : {detections['detection_masks_reframed'][0]}")
                print(f" original : {detections['detection_masks'][0]}")

                exit()             

                boxes, classes, scores = postprocessing(item['np_image'], detections['detection_boxes'], detections['detection_classes'],detections['detection_scores'],1e-2)


                
                for box, cls, score in zip (boxes,
                                            classes,
                                            scores):
                    # convert to x,y,w,w for coco

                    c_x = float(box[0])
                    c_y = float(box[1])
                    w = float(box[2] - box[0] +1 )
                    h = float(box[3] - box[1] +1 )

                    results.append({'image_id':item['imageId'],
                                    'category_id': int(cls),
                                    'bbox':[c_x,c_y,w,h],
                                    'score': float(score)})
                total_image = len(images)
            print('average time pro batch: {:4.1f} ms'.format((elapsed_time[-len(images):].mean()) * 1000 ))

            ## save predicted annotation
            print(f"Total evaluate {len(total_image)} \t")
            print(f"save results in to json!")
            save_performance('prediction', json_data=results, file_name= self.__model_name +'.json')
            results.clear()
        print(f'total time {(sum(elapsed_time)/len(elapsed_time)*1000)}')
        print('After all Evaluation FPS {:4.1f} ms '.format(1000/(sum(elapsed_time)/len(elapsed_time)*1000)))
  

    """
        Run detection on each image an write result into Json file
    """
    def generate_detection_results_ssd(self):
        elapsed_time = []
        results = []
        total_image = 0
        batch_count = 0

        for images in load_img_from_folder(self.__path_to_images, validation_split=1, batch_size=self.__batch_size, mAP=True):
            # convert images to be a tensor
            batch_count = batch_count + 1
            print(f"run evaluation for batch {batch_count}. \t")
            for item in images:
                input_tensort = tf.convert_to_tensor(item['np_image'])
                input_tensort = input_tensort[tf.newaxis,...]
                try:
                    start_time = time.time()

                    detections = self.__model(input_tensort)

                    end_time = time.time()

                    num_detections = int(detections.pop('num_detections'))
                    detections = {key: value[0, :num_detections].numpy()
                                for key, value in detections.items()}
                    detections['num_detections'] = num_detections 
                    # convert detection  classes to  numpy int
                    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

                    elapsed_time = np.append(elapsed_time, end_time - start_time)
                except :
                    continue
                
                boxes, classes, scores = postprocessing(item['np_image'], detections['detection_boxes'], detections['detection_classes'],detections['detection_scores'],1e-2)              
                
                for box, cls, score in zip (boxes,
                                            classes,
                                            scores):
                    # convert to x,y,w,w for coco

                    c_x = float(box[0])
                    c_y = float(box[1])
                    w = float(box[2] - box[0] +1 )
                    h = float(box[3] - box[1] +1 )
                                        
                    results.append({'image_id':item['imageId'],
                                    'category_id': int(cls),
                                    'bbox':[c_x,c_y,w,h],
                                    'score': float(score)})
            
            total_image =  total_image + len(images)
            print('average time pro batch: {:4.1f} ms'.format((elapsed_time[-len(images):].mean()) * 1000 ))

            ## save predicted annotation 
            print(f"Total evaluate {total_image} \t")
            print(f"save results in to json!")
            save_performance('prediction', json_data=results, file_name= self.__model_name +'.json')
            results.clear()

        print(f'total time {(sum(elapsed_time)/len(elapsed_time)*1000)}')
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



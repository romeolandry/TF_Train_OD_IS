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
                 path_to_annotaions,
                 batch_size=32):
        self.__path_to_images = path_to_images
        self.__model = model
        self.__model_name = model_name
        self.__path_to_annotations = path_to_annotaions
        self.__batch_size = batch_size

        ## create category index for coco 
        

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
            # detection_classes should be ints.
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

    def generate_detection_results(self):
        """
        Run detection on each image an write result into Json file
        """
        elapsed_time = []
        results = []
        total_image = 0
        batch_count = 0

        for images in load_img_from_folder(self.__path_to_images, bacth_size=self.__batch_size, mAP=True):
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

                for boxe, classe, score in zip (detections['detection_boxes'],
                                                detections['detection_classes'],
                                                detections['detection_scores']):
                    x = float(boxe[0])
                    y = float(boxe[1])
                    w = float(boxe[2]- boxe[0] + 1)
                    h = float(boxe[3]- boxe[1] + 1)
                    results.append({'image_id':item['imageId'],
                                    'category_id': int(classe),
                                    'bbox':[x, y, w, h],
                                    'score': float(score)})
                    print("detection")
                    print(boxe)
                    print("in results")
                    print(f" x: {x}, y: {y}, w: {w}, h: {h}")
                    break
                total_image = len(images)
            exit()
            print('average FPS pro batch: {:4.1f}ms'.format((elapsed_time[-len(images):].mean()) * self.__batch_size ))

            ## save predicted annotation
            print(f"Total evaluate {len(images)} \t")
            print(f"save results in to json!")
            save_perfromance('prediction', json_daten=results, file_name= self.__model_name +'.json')
            results.clear()
        totale_time = total_image * self.__batch_size / elapsed_time.sum()
        print(f'total time {totale_time}')
        print('Throughput: {:.0f} images/s'.format(total_image / elapsed_time.sum()))


    def metric_with_api():
        pass

    def COCO_mAP_bbox(self):
        cocoGt = COCO(self.__path_to_annotations)
        cocoDt = cocoGt.loadRes(os.path.join(PATH_PERFORMANCE_INFER,self.__model_name + '.json'))

        imgIds = sorted(cocoGt.getImgIds())

        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        
        cocoEval.params.imgIds = imgIds

        cocoEval.evaluate()
        cocoEval.accumulate()

        print(cocoEval.summarize())



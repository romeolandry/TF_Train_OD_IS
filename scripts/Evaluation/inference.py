import os
import sys
import time
import click
import matplotlib.pyplot as plt
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import tensorflow as tf

sys.path.append(os.path.abspath(os.curdir))

from configs.run_config import *
from scripts.api_scrpit import *
from scripts.Evaluation.utils import *

from scripts.api_scrpit import label_map_util
from scripts.api_scrpit import visualization_utils as viz_utils


# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)



class Inefrence :
    def __init__(self,
                 path_to_images,
                 path_to_labels,
                 model,
                 model_name):
        self.__path_to_images = path_to_images
        self.__path_to_labels = path_to_labels
        self.__model = model
        self.__images_name_prefix = model_name

        ## create category index for coco 
        self.__category_index = label_map_util.create_category_index_from_labelmap(self.__path_to_labels,use_display_name=True)


    def infernce_images_from_dir(self, number_of_images=None):

        images_list = []
        print(f"loading {number_of_images} images from {self.__path_to_images}")
        images_list = load_img_from_folder_for_infer(self.__path_to_images,number_of_images)
        i = 0
        for image_np in images_list:
            # convert images to be a tensor
            input_tensor = tf.convert_to_tensor(image_np)
            input_tensor = input_tensor[tf.newaxis, ...]
            detections = self.__model(input_tensor)
            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
                          for key, value in detections.items()}

            detections['num_detections'] = num_detections


            # detection_classes should be ints.
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
            image_np_with_detections = image_np.copy()

            viz_utils.visualize_boxes_and_labels_on_image_array(image_np_with_detections,
                                                                detections['detection_boxes'],
                                                                detections['detection_classes'],
                                                                detections['detection_scores'],
                                                                self.__category_index,
                                                                use_normalized_coordinates=True,
                                                                max_boxes_to_draw=200,
                                                                min_score_thresh=.30,
                                                                agnostic_mode=False)
                

            plt.figure()
            plt.imshow(image_np_with_detections)
            if not os.path.isdir(PATH_DIR_IMAGE_INF):
                os.mkdir(PATH_DIR_IMAGE_INF)
            plt.savefig(os.path.join(PATH_DIR_IMAGE_INF,self.__images_name_prefix + "_" + str(i)))
            i +=1
        print('Done')
            #plt.show()

    @tf.function
    def detect_fn(self,image):

        """Detect objects in image."""
        image, shapes = self.__model.preprocess(image)
        prediction_dict = self.__model.predict(image, shapes)
        detections = self.__model.postprocess(prediction_dict, shapes)

        return detections, prediction_dict, tf.reshape(shapes, [-1])


    def inference_from_wedcam_with_checkpoint(self):

        # build the model 
        #self.build_detection_model()

        cap = cv2.VideoCapture(0)

        while True:
            # Read frame from camera
            ret, image_np = cap.read()

            # expand image to have shape :[1, None, None,3]
            image_np_expanded = np.expand_dims(image_np, axis=0)

            input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0),dtype=tf.float32)
            detections, predictions_dict, shapes = self.detect_fn(input_tensor)

            label_id_offset = 1
            image_np_with_detections = image_np.copy()

            viz_utils.visualize_boxes_and_labels_on_image_array(image_np_with_detections,
                                                                detections['detection_boxes'][0].numpy(),
                                                                (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
                                                                detections['detection_scores'][0].numpy(),
                                                                self.__category_index,
                                                                use_normalized_coordinates=True,
                                                                max_boxes_to_draw=200,
                                                                min_score_thresh=.30,
                                                                agnostic_mode=False)


            # Display output
            cv2.imshow('object detection', cv2.resize(image_np_with_detections, (800, 600)))

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

    def inference_from_wedcam_with_checkpoint2(self):
    
        # build the model 
        #self.build_detection_model()

        cap = cv2.VideoCapture(0)

        while True:
            # Read frame from camera
            ret, image_np = cap.read()
            input_tensor = tf.convert_to_tensor(image_np)
            input_tensor = input_tensor[tf.newaxis, ...]
            # expand image to have shape :[1, None, None,3]
            #image_np_expanded = np.expand_dims(image_np, axis=0)

            #input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0),dtype=tf.float32)


            #detections, predictions_dict, shapes = self.detect_fn(input_tensor)

            detections = self.__model(input_tensor)

            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
                          for key, value in detections.items()}

            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
            
            detections['num_detections'] = num_detections

            label_id_offset = 1
            image_np_with_detections = image_np.copy()

            viz_utils.visualize_boxes_and_labels_on_image_array(image_np_with_detections,
                                                                detections['detection_boxes'][0],
                                                                detections['detection_classes'][0],
                                                                detections['detection_scores'][0],
                                                                self.__category_index,
                                                                use_normalized_coordinates=True,
                                                                max_boxes_to_draw=200,
                                                                min_score_thresh=.30,
                                                                agnostic_mode=False)


            # Display output
            cv2.imshow('object detection', cv2.resize(image_np_with_detections, (800, 600)))

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
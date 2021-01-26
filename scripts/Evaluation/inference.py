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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sys.path.append(os.path.abspath(os.curdir))

from configs.run_config import *
#from scripts.api_scrpit import *
from scripts.Evaluation.utils import *

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

#from scripts.api_scrpit import label_map_util
#from scripts.api_scrpit import visualization_utils as viz_utils


# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)



class Inference :
    def __init__(self,
                 path_to_images,
                 path_to_labels,
                 model,
                 model_name="output",
                 model_image_size = None):
        self.__path_to_images = path_to_images
        self.__path_to_labels = path_to_labels
        self.__model = model
        self.__images_name_prefix = model_name
        self.__model_image_size = model_image_size

        ## create category index for coco 
        self.__category_index = label_map_util.create_category_index_from_labelmap(self.__path_to_labels,use_display_name=True)

    ''' 
        Using SSD-resnet50v2 to apply inference on a list of image
        The image will be progressively load and inference
        the output images will be save in to 'images_inferences'
    '''
    def ssd_inference_image (self, number_of_images=None):

        images_list = []
        print(f"loading {number_of_images} images from {self.__path_to_images}")
        images_list = load_img_from_folder_for_infer(self.__path_to_images,number_of_images, image_size=self.__model_image_size)
        i = 0
        for image_np in images_list:
            # convert images to be a tensor
            input_tensor = tf.convert_to_tensor(image_np)
            input_tensor = input_tensor[tf.newaxis, ...]

            input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.uint8)
            # Apply inference 
            detections = self.__model(input_tensor)

            ## convert all output to a numpy array
            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
                          for key, value in detections.items()}

            detections['num_detections'] = num_detections
            # detection_classes should be int64.
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

    '''
        Using Mask-RCNN-inception_resnet_v2 to apply inference on a list of image
        The image will be progressively load and inference
        the output images will be save in to 'images_inferences'
    '''
    def mask_inference_image(self, number_of_images=None):
    
        images_list = []
        print(f"loading {number_of_images} images from {self.__path_to_images}")
        images_list = load_img_from_folder_for_infer(self.__path_to_images,number_of_images,image_size=self.__model_image_size)
        i = 0
        for image_np in images_list:
            # convert images to be a tensor
            input_tensor = tf.convert_to_tensor(image_np)
            input_tensor = input_tensor[tf.newaxis, ...]
            input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.uint8)
           
            label_id_offset = 1
            image_np_with_detections = image_np.copy()
            detections = self.__model(input_tensor)
            if 'detection_masks' in detections:
                detection_masks = tf.convert_to_tensor(detections['detection_masks'][0])
                detection_boxes = tf.convert_to_tensor(detections['detection_boxes'][0])

                # Reframe the the bbox mask to the image size.

                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks, detection_boxes,image_np.shape[0], image_np.shape[1])
                detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,tf.uint8)
                detections['detection_masks_reframed'] = detection_masks_reframed.numpy()

            viz_utils.visualize_boxes_and_labels_on_image_array(image_np_with_detections,
                                                detections['detection_boxes'][0].numpy(),
                                                (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
                                                detections['detection_scores'][0].numpy(),
                                                self.__category_index,
                                                instance_masks=detections.get('detection_masks_reframed',None),
                                                use_normalized_coordinates=True,
                                                line_thickness=8)
            
            plt.figure()
            plt.imshow(image_np_with_detections)
            plt.show()
            if not os.path.isdir(PATH_DIR_IMAGE_INF):
                os.mkdir(PATH_DIR_IMAGE_INF)
            plt.savefig(os.path.join(PATH_DIR_IMAGE_INF,self.__images_name_prefix + "_" + str(i)))
            i +=1
        print('Done')
    
    
    @tf.function
    def detect_fn(self,image):

        """Detect objects in image."""
        image, shapes = self.__model.preprocess(image)
        prediction_dict = self.__model.predict(image, shapes)
        detections = self.__model.postprocess(prediction_dict, shapes)

        return detections, prediction_dict, tf.reshape(shapes, [-1])


    '''
        Using SSD-resnet50 v2  to apply inference on a list of image
        The image will be progressively load and inference
        the output images will be save in to 'images_inferences'
    '''
    def ssd_inference_webcam(self):

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


    '''
        Using SSD-resnet50 v2  to apply inference on a list of image
        The image will be progressively load and inference
        the output images will be save in to 'images_inferences'
    '''
    def mask_inference_webcam(self, number_of_images=None):

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
            
            if 'detection_masks' in detections:
    
                detection_masks = tf.convert_to_tensor(detections['detection_masks'][0])
                detection_boxes = tf.convert_to_tensor(detections['detection_boxes'][0])

                # Reframe the the bbox mask to the image size.

                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks, detection_boxes,image_np.shape[0], image_np.shape[1])
                detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,tf.uint8)
                detections['detection_masks_reframed'] = detection_masks_reframed.numpy()

            viz_utils.visualize_boxes_and_labels_on_image_array(image_np_with_detections,
                                                detections['detection_boxes'][0].numpy(),
                                                (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
                                                detections['detection_scores'][0].numpy(),
                                                self.__category_index,
                                                instance_masks=detections.get('detection_masks_reframed',None),
                                                use_normalized_coordinates=True,
                                                max_boxes_to_draw= 200,
                                                min_score_thresh=.30,
                                                agnostic_mode=False,
                                                line_thickness=8)


            # Display output
            cv2.imshow('Instance Segmentation', cv2.resize(image_np_with_detections, (800, 600)))

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()


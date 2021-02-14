import os
import sys
import time
import click
import matplotlib.pyplot as plt
import cv2 as cv

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import tensorflow as tf
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sys.path.append(os.path.abspath(os.curdir))

from configs.run_config import *
#from scripts.api_scrpit import *
from scripts.Evaluation.utils import *

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from scripts.Evaluation.utils import read_label_txt, set_input_camera

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# set input output name
inputs = ["input_tensor:0"]
outputs = ["Identity:0","Identity_1:0","Identity_2:0","Identity_3:0","Identity_4:0","Identity_5:0","Identity_6:0","Identity_7:0"]

def wrap_frozen_graph(graph_def, inputs, outputs, print_graph=False):
    def _imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")

    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph

    print("-" * 50)
    print("Frozen model layers: ")
    layers = [op.name for op in import_graph.get_operations()]
    if print_graph == True:
        for layer in layers:
            print(layer)
    print("-" * 50)

    return wrapped_import.prune(
        tf.nest.map_structure(import_graph.as_graph_element, inputs),
        tf.nest.map_structure(import_graph.as_graph_element, outputs))


"""
    Post preprocessing ssd
"""
def postprocessing(img, boxes,classes, scores, th=0.1):
    w, h, _ = img.shape
   
    output_boxes = boxes #* np.array([w,h,w,h])
     #output_boxes = output_boxes.astype(np.int32)
    #output_boxes = output_boxes[:,[1,0,3,2]] # wrap x's y's each box[x0,y0,x1,y1]=> box[y0,x0,y1,x1]
    output_confs = scores
    output_cls =classes.astype(np.int32)

    # return bboxes with confidence score above threshold
    mask = np.where(output_confs >= th)
    return output_boxes[mask], output_cls[mask], output_confs[mask],w,h


class Inference :
    def __init__(self,
                 path_to_images,
                 path_to_labels,
                 model,
                 model_name="output",
                 model_image_size = 640,
                 threshold=0.5):
        self.__path_to_images = path_to_images
        self.__path_to_labels = path_to_labels
        self.__model = model
        self.__images_name_prefix = model_name
        self.__model_image_size = model_image_size
        self.__threshold = threshold

        ## create category index for coco 
        self.__category_index = label_map_util.create_category_index_from_labelmap(self.__path_to_labels,use_display_name=True)

    
    '''
        draw bbox on image
    '''
    def visualize_bbox(self,image,score,bbox,classId):
        
        width = image.shape[0]
        height = image.shape[1]

        categories = read_label_txt(PATH_TO_LABELS_TEXT)

        # get class text
        label = categories[classId]
        # get class text
        scored_label = label + ' ' + format(score * 100, '.2f')+ '%'
        x = bbox[1] * height
        y = bbox[0] * width
        right = bbox[3] * height
        bottom = bbox[2] * width
        font = cv.FONT_HERSHEY_COMPLEX

        cv.rectangle(image, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)
        
        cv.putText(image, scored_label, (int(x)+10, int(y)+20),font, 1, (0, 255, 0), thickness=1)
        return image
        


    ''' 
        Using SSD-savedModel to apply inference on a list of image
        The image will be progressively load and inference
        the output images will be save in to 'images_inferences'
        ** require Tf-Object-Detection-AIP since the his visualization tools was used,
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
        Using SSD-resnet50v to apply inference on image with openCV
        input model a  graph that was read from freezed modell
        the output images will be save in to 'images_inferences'

        As Tf 2.x don't use Session and Graph anymore
        we use TF 1.X for inference
    '''
    def ssd_inference_freezed_model(self):
        # read and Preprocess image 
        img = cv.imread(self.__path_to_images)
        
        inp = cv.resize(img, (self.__model_image_size[0], self.__model_image_size[1]))
        inp = inp[:, :, [2, 1, 0]]  # BGR2RGB
        # read frozen Graph       
        elapsed_time = 0
        with tf.compat.v1.Session() as sess:
            sess.graph.as_default()
            start =time.time()
            tf.import_graph_def(self.__model, name='')

            frozen_func = wrap_frozen_graph(graph_def=self.__model,
                                            inputs=inputs,
                                            outputs=outputs,
                                            print_graph=False)
            end = time.time()
            click.echo(click.style(f"\n Wrapped the freezed model for inference in   {end-start} seconds. \n", bold=True, fg='green'))

            # Apply the model
            # Identity_5:0 => num_detections
            # Identity_4 => detection_scores
            # Identity_2:0 => detection_classes
            # Identity_1:0 => detection_boxes
            out = sess.run([sess.graph.get_tensor_by_name('Identity_5:0'),
                    sess.graph.get_tensor_by_name('Identity_4:0'),
                    sess.graph.get_tensor_by_name('Identity_2:0'),
                    sess.graph.get_tensor_by_name('Identity_1:0')],
                   feed_dict={'input_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

            # Visualize detected bounding boxes.
            num_detections = int(out[0][0])
            for i in range(num_detections):
                classId = int(out[2][0][i])
                score = float(out[1][0][i])
                bbox = [float(v) for v in out[3][0][i]]

                if score > self.__threshold:
                    img = self. visualize_bbox(img,score,bbox,classId)

        if not os.path.isdir(PATH_DIR_IMAGE_INF):
            os.mkdir(PATH_DIR_IMAGE_INF)
        
        img_path = os.path.join(PATH_DIR_IMAGE_INF,self.__images_name_prefix +'.png') 
        print(img_path)
        cv.imwrite(img_path,img)
            
        cv.imshow('TensorFlow SSD-ResNet', img)
        cv.waitKey(1)


    
    def ssd_inference_webcam_freezed_model(self,camera_input,camera_width,camera_height):
       cap = set_input_camera(camera_input,camera_width,camera_height)
       with tf.compat.v1.Session() as sess:
            sess.graph.as_default()
            start =time.time()
            tf.import_graph_def(self.__model, name='')

            frozen_func = wrap_frozen_graph(graph_def=self.__model,
                                            inputs=inputs,
                                            outputs=outputs,
                                            print_graph=False)
            end = time.time()
            click.echo(click.style(f"\n Wrapped the freezed model for inference in   {end-start} seconds. \n", bold=True, fg='green'))
            
            while True:
                cap.read()
                success, frame = cap.read()
                if not success:
                    break           
            
                height = frame.shape[0]
                width = frame.shape[1]
                img = np.array(frame)

                img_to_infer = cv.resize(img, (self.__model_image_size[0],self.__model_image_size[1]), interpolation=cv2.INTER_CUBIC)
                           
                
                # Apply the prediction
                # Identity_5:0 => num_detections
                # Identity_4 => detection_scores
                # Identity_2:0 => detection_classes
                # Identity_1:0 => detection_boxes
                out = sess.run([sess.graph.get_tensor_by_name('Identity_5:0'),
                        sess.graph.get_tensor_by_name('Identity_4:0'),
                        sess.graph.get_tensor_by_name('Identity_2:0'),
                        sess.graph.get_tensor_by_name('Identity_1:0')],
                    feed_dict={'input_tensor:0': img_to_infer.reshape(1, img_to_infer.shape[0], img_to_infer.shape[1], 3)})
                
                # Visualize detected bounding boxes.
                num_detections = int(out[0][0])
                for i in range(num_detections):
                    classId = int(out[2][0][i])
                    score = float(out[1][0][i])
                    bbox = [float(v) for v in out[3][0][i]]

                    if score > self.__threshold:
                        img = self. visualize_bbox(img,score,bbox,classId)

                print(f" This model compute {int(cap.get(cv2.CAP_PROP_FPS))} FPS ")
                cv.imshow(self.__images_name_prefix,img)
                cv.waitKey(1)

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
                
                print(f"detect bounding box {detections['detection_boxes'][0]}")
                boxes, classes, scores,width,height = postprocessing(item['np_image'], detections['detection_boxes'], detections['detection_classes'],detections['detection_scores'],1e-2)              
                
                print(f" after postprocessing {boxes}")
                
                for box, cls, score in zip (boxes,
                                            classes,
                                            scores):
                    # convert to x,y,w,w for coco

                    ymin, xmin, ymax, xmax = box
                    
                    c_x = float(box[0])
                    c_y = float(box[1])
                    w =  float(box[2]-box[0])
                    h =  float(box[3]-box[1])
                    
                    print(f" coco coordinate: c_x  {c_x}: c_y: {c_y} w: {w}: h: {h}")

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
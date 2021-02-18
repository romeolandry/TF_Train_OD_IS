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

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

from configs.run_config import *
#from scripts.api_scrpit import *
from scripts.Evaluation.utils import *

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

from scripts.Evaluation.utils import read_label_txt, set_input_camera

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# set input output name
inputs = ["input_tensor:0"]
# mask have 23 outputs node 
outputs = ["Identity:0"]
outputs.extend([f"Identity_{i}:0" for i in range(1,23)])

def wrap_frozen_graph(graph_def, inputs, outputs, print_graph=False):
    def _imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")

    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph

    print("-" * 50)
    print("Frozen model layers: ")
    layers = [op.name for op in import_graph.get_operations()]
    if print_graph == True:
        print("Output layers")
        for layer in layers:
           print(layer)
    print("-" * 50)

    return wrapped_import.prune(
        tf.nest.map_structure(import_graph.as_graph_element, inputs),
        tf.nest.map_structure(import_graph.as_graph_element, outputs))




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
        draw bbox and mask on image
    '''
    def visualize_bbox_mask(self,image,score,bbox,mask,classId):
        
        width = image.shape[0]
        height = image.shape[1]

        img = image.copy()

        categories = read_label_txt(PATH_TO_LABELS_TEXT)

        # get class text
        label = categories[classId]
        # get class text
        scored_label = label + ' ' + format(score * 100, '.2f')+ '%'

        bbox = bbox * np.array([width, height, width, height])
        y, x, bottom, right = bbox.astype("int")
        # mask
        mask = cv.resize(mask,(width,height),interpolation=cv.INTER_NEAREST)
        mask = (mask > self.__threshold)
        # extract ROI of image
        roi =  img[y:bottom, x:right]

        print(f"mask: {mask.shape}")

        print(f"roi: {roi.shape}")

        roi = roi[mask]

        blended = ((0.4 * (255,0,255)) + (0.6 * roi)).astype("uint8")
        img[y:bottom, x:right][mask] = blended

        font = cv.FONT_HERSHEY_COMPLEX

        cv.rectangle(img, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)

        cv.putText(img, scored_label, (int(x)+10, int(y)+20),font, 1, (0, 255, 0), thickness=1)

        return img

    
    
    def mask_inference_image_cv2 (self, number_of_images=None):
        
        # read and Preprocess image 
        img = cv.imread(self.__path_to_images)
        
        image_np = cv.resize(img, (self.__model_image_size[0], self.__model_image_size[1]))
        image_np = image_np[:, :, [2, 1, 0]]  # BGR2RGB
        # read frozen Graph       
        elapsed_time = 0

        # convert images to be a tensor
        input_tensor = tf.convert_to_tensor(image_np)
        input_tensor = input_tensor[tf.newaxis, ...]

        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.uint8)
        # Apply inference 
        detections = self.__model(input_tensor)

        ## convert all output to a numpy array
        num_detections = int(detections.pop('num_detections'))

        exit(num_detections)
        detections = {key: value[0, :num_detections].numpy()
                        for key, value in detections.items()}

        detections['num_detections'] = num_detections
        # detection_classes should be int64.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        
        # Visualize detected bounding boxes.        
        for i in range(num_detections):
            classId = detections['detection_classes'][i]
            score = detections['detection_scores'][i]
            bbox = [float(v) for v in detections['detection_boxes'][i]]


            if score > self.__threshold:
                img = self.visualize_bbox(img,score,bbox,classId)

        img_path = os.path.join(PATH_DIR_IMAGE_INF,self.__images_name_prefix + "savedmodel_cv2_test.png")
        cv.imwrite(img_path,img)
            
        cv.imshow('TensorFlow SSD-ResNet', img)
        cv.waitKey(0)
        print('Done')


    ''' 
        Using mask r-cnn to apply inference on image with openCV
        input model a  graph that was read from freezed modell
        the output images will be save in to 'images_inferences'

        As Tf 2.x don't use Session and Graph anymore
        we use TF 1.X for inference
    '''
    def mask_inference_freezed_model(self):
        # read and Preprocess image 
        img = cv.imread(self.__path_to_images)
        
        img = cv.resize(img, (self.__model_image_size[0], self.__model_image_size[1]))
        img = img[:, :, [2, 1, 0]]  # BGR2RGB
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
            # Identity_12:0 => num_detections
            # Identity_8 => detection_scores
            # Identity_5:0 => detection_classes
            # Identity_4:0 => detection_boxes
            # Identity_6:0 => detection_masks

            out = sess.run([sess.graph.get_tensor_by_name('Identity_12:0'),
                    sess.graph.get_tensor_by_name('Identity_4:0'),
                    sess.graph.get_tensor_by_name('Identity_5:0'),
                    sess.graph.get_tensor_by_name('Identity_8:0'),
                    sess.graph.get_tensor_by_name('Identity_6:0')],
                   feed_dict={'input_tensor:0': img.reshape(1, img.shape[0], img.shape[1], 3)})

            
            # Visualize detected bounding boxes.
            num_detections = int(out[0][0])
            for i in range(num_detections):
                classId = int(out[2][0][i])
                score = float(out[3][0][i])
                bbox = [float(v) for v in out[1][0][i]]
                mask =  out[4][0][i]
                if score > self.__threshold:
                    img = self. visualize_bbox_mask(img,score,bbox,mask,classId)

        if not os.path.isdir(PATH_DIR_IMAGE_INF):
            os.mkdir(PATH_DIR_IMAGE_INF)
        
        img_path = os.path.join(PATH_DIR_IMAGE_INF,self.__images_name_prefix +'.png') 
        print(img_path)
        cv.imwrite(img_path,img)
            
        cv.imshow('mask freezed graph', img)
        cv.waitKey(1)


    
    def mask_inference_webcam_freezed_model(self,camera_input,camera_width,camera_height):
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
        
        # read and Preprocess image 
        img = cv.imread(self.__path_to_images)
        
        image_np = cv.resize(img, (self.__model_image_size[0], self.__model_image_size[1]))
        image_np = image_np[:, :, [2, 1, 0]]  # BGR2RGB

        # convert images to be a tensor
        input_tensor = tf.convert_to_tensor(image_np)
        input_tensor = input_tensor[tf.newaxis, ...]
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.uint8)

        image_np_with_detections = image_np.copy()

        detections = self.__model(input_tensor)

        if 'detection_masks' in detections:
            detection_masks = tf.convert_to_tensor(detections['detection_masks'][0])
            detection_boxes = tf.convert_to_tensor(detections['detection_boxes'][0])
            # Reframe the the bbox mask to the image size.
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks, detection_boxes,image_np.shape[0], image_np.shape[1])
            detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,tf.uint8)
            detections['detection_masks_reframed'] = detection_masks_reframed.numpy()
            
        label_id_offset=1
        viz_utils.visualize_boxes_and_labels_on_image_array(image_np_with_detections,
                                            detections['detection_boxes'][0].numpy(),
                                            (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
                                            detections['detection_scores'][0].numpy(),
                                            self.__category_index,
                                            instance_masks=detections.get('detection_masks_reframed',None),
                                            use_normalized_coordinates=True,
                                            line_thickness=8)

        if not os.path.isdir(PATH_DIR_IMAGE_INF):
            os.mkdir(PATH_DIR_IMAGE_INF)
        
        img_path = os.path.join(PATH_DIR_IMAGE_INF,self.__images_name_prefix +'.png') 

        cv.imwrite(img_path,image_np_with_detections)
        print("Done!")
    
    
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


'''bind mask and Box'''
def transform_detection_to_cocoresult(image,boxes,masks,classes,scores):
    
    boxes = boxes.numpy()
    masks = masks.numpy()
    scores = scores.numpy()
    classes = classes.numpy()
    
    assert boxes.shape[0] == masks.shape[0] == classes.shape[0] == scores.shape[0]

    h,w = image['np_image'].shape[:2]

    if boxes is None:
        return []

    results = []

    
    for i in range(boxes.shape[0]):
        classId = classes[i]
        score = scores[i]
        bbox = np.around(boxes[i],1)
        # normalise 
        bbox = bbox * np.array([w,h,w,h])
        boxW = bbox[3]-bbox[1]
        boxH = bbox[2]-bbox[0]
        mask = cv.resize(masks[i], (boxW, boxH), interpolation=cv.INTER_NEAREST)
        mask = np.uint8(mask)     
        
        result = {
            "image_id":image['imageId'],
            "category_id": int(classId),
            "bbox":[bbox[1],bbox[0],boxW,boxH],
            "score": float(score),
            "segmentation": maskUtils.encode(np.asfortranarray(mask))
        }
        results.append(result)
    return results      

"""
    Post preprocessing mask
"""
def postprocessing(img, boxes,classes, scores,mask, th=0.1):
    
    boxes = boxes.numpy()
    masks = masks.numpy()
    scores = scores.numpy()
    classes = classes.numpy()

    
    assert boxes.shape[0] == masks.shape[0] == classes.shape[0] == scores.shape[0]
    h,w= img.shape[:2]

    categories = read_label_txt(PATH_TO_LABELS_TEXT)

    img = img.copy()

    for i in range(boxes.shape[0]):
        classId = classes[i]
        score = scores[i]
        # get class text
        label = categories[classId]
        # get class text
        scored_label = label + ' ' + format(score * 100, '.2f')+ '%'

        if not np.any(boxes[i]):
            # skip instance that has no bbox
            continue

        font = cv.FONT_HERSHEY_COMPLEX

        box = boxes[i]* np.array([w,h,w,h])
        (startY,startX,endY,endX) = box,astype("int") # top,left right, bottom

        cv.rectangle(img, (startX, startX), (int(endY), int(endX)), (125, 255, 51), thickness=2)
        cv.putText(img, scored_label, (int(startX)+10, int(startX)+20),font, 1, (0, 255, 0), thickness=1)

        boxW = endX - startX
        boxH = endY - startY

        mask = cv.resize(masks[i], (boxW, boxH), interpolation=cv.INTER_NEAREST)

        mask = (mask > .05)

        roi = img[startY:endY, startX:endX]
        

        roi = [mask]

        blended = ((0.4 * (255,0,0)) + (0.6 * roi)).astype("uint8")

        roi = img[startY:endY, startX:endX][mask]= blended


        




    

class Evaluation:
    def __init__(self,
                 path_to_images,
                 model,
                 model_name,
                 path_to_annotations,
                 batch_size=32):
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
               
                try:
                    # convert images to be a tensor
                    input_tensort = tf.convert_to_tensor(item['np_image'])
                    input_tensort = input_tensort[tf.newaxis, ...]

                    #input_tensort = tf.convert_to_tensor(np.expand_dims(item['np_image'], 0), dtype=tf.uint8)
               
                    start_time = time.time()
                    detections = self.__model(input_tensort)
                    end_time = time.time()
                except Exception as e:
                    continue

                elapsed_time = np.append(elapsed_time, end_time - start_time)

                boxes = detections['detection_boxes']
                classes = detections['detection_classes']
                scores = detections['detection_scores']
                masks  = detections['detection_masks']

                result = transform_detection_to_cocoresult(item,boxes[0],masks[0],classes[0],scores[0])
                
                results.extend(result)
            total_image = total_image + len(images)
            print('average time pro batch: {:4.1f} ms'.format((elapsed_time[-len(images):].mean()) * 1000 ))            
            print(f"Total evaluate {total_image} \t")
        
        ## save predicted annotation
        print(f"save results in to json!")
        save_performance('prediction', json_data=results, file_name= self.__model_name +'.json')
            
        print(f'total time {(sum(elapsed_time)/len(elapsed_time)*1000)}')
        print('After all Evaluation FPS {:4.1f} ms '.format(1000/(sum(elapsed_time)/len(elapsed_time)*1000)))
  

    
    def COCO_process_mAP(self,type):
        cocoGt = COCO(self.__path_to_annotations)
        cocoDt = cocoGt.loadRes(os.path.join(PATH_PERFORMANCE_INFER,self.__model_name + '.json'))
        
        imgIds = sorted(cocoGt.getImgIds())
        cocoEval.params.imgIds = imgIds
        
        cocoEval = COCOeval(cocoGt, cocoDt, type)        
        
        cocoEval.evaluate()
        cocoEval.accumulate()

        print(cocoEval.summarize())
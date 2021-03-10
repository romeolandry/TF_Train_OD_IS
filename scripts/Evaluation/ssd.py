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

sys.path.append(os.path.abspath(os.curdir))

from configs.run_config import *
#from scripts.api_scrpit import *
from scripts.Evaluation.utils import *

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


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
    visualize bbox
"""
def viewer(img, boxes,classes, scores, th=0.1):
    
    boxes = boxes.numpy()
    masks = masks.numpy()
    scores = scores.numpy()
    classes = classes.numpy()

    
    assert boxes.shape[0]  == classes.shape[0] == scores.shape[0]
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

    return img


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
        # self.__category_index = label_map_util.create_category_index_from_labelmap(self.__path_to_labels,use_display_name=True)

    
    '''
        draw bbox on image
    '''
    def visualize_bbox(self,image,score,bbox,classId):
        
        width = image.shape[0]
        height = image.shape[1]

        categories = read_label_txt(PATH_TO_LABELS_TEXT)

        lastbbox= None
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
        if lastbbox == bbox[1]:
            cv.putText(image, scored_label, (int(x)+30, int(y)+50),font, 1, (0, 255, 0), thickness=1)
        else:
            cv.putText(image, scored_label, (int(x)+10, int(y)+20),font, 1, (0, 255, 0), thickness=1)
        lastbbox = bbox[1]
        return image

    ''' 
        Using SSD-savedModel to apply inference on one image
    '''
    def ssd_inference_image_cv2 (self, number_of_images=None):
    
        # read and Preprocess image 
        img = cv.imread(self.__path_to_images)
        
        image_np = cv.resize(img, (self.__model_image_size[0], self.__model_image_size[1]))
        image_np = image_np[:, :, [2, 1, 0]]  # BGR2RGB
       
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

        
        # Visualize detected bounding boxes.        
        for i in range(num_detections):
            classId = detections['detection_classes'][i]
            score = detections['detection_scores'][i]
            bbox = [float(v) for v in detections['detection_boxes'][i]]


            if score > self.__threshold:
                img = self. visualize_bbox(img,score,bbox,classId)

        img_path = os.path.join(PATH_DIR_IMAGE_INF,self.__images_name_prefix + "_savedmodel_ssd_cv2.png")
        cv.imwrite(img_path,img)
            
        cv.imshow(self.__images_name_prefix, img)
        cv.waitKey(1)
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
        
        img_path = os.path.join(PATH_DIR_IMAGE_INF,self.__images_name_prefix +'_ssd.png') 
        print(img_path)
        cv.imwrite(img_path,img)
            
        cv.imshow(self.__images_name_prefix, img)
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
                cv.imshow(self.__images_name_prefix,img)
                cv.waitKey(1)
   
    '''
        Using SSD-resnet50 v2  to apply inference on a list of image
        The image will be progressively load and inference
        the output images will be save in to 'images_inferences'
    '''
    ''' def ssd_inference_webcam(self):

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

'''bind mask and Box'''
def transform_detection_to_cocoresult(image_id,
                                      image_width,
                                      image_height,
                                      boxes,
                                      classes,
                                      scores):
    
    assert boxes.shape[0] == classes.shape[0] == scores.shape[0]
    if boxes is None:
        return []

    results = []    
    for i in range(boxes.shape[0]):
        classId = classes[i]
        score = scores[i]
        bbox = np.around(boxes[i],1)

        y1,x1,y2,x2 = list(bbox)

        # normalized coco coordinate 
        #bbox = bbox * np.array([w,h,w,h])
        # [ymin,xmin,ymax,xmax] [bbox[1],bbox[0],boxW,boxH]
        #boxW = bbox[3]-bbox[1] + 1
        #boxH = bbox[2]-bbox[0] + 1
        #[xmin,ymin,xmax,ymax] [bbox[0],bbox[1],boxW,boxH]
        #boxW = bbox[2]-bbox[0] + 1
        #boxH = bbox[3]-bbox[1] + 1
        x1 = x1 * image_width
        y1 = y1 * image_height
        width = (x2-x1)* image_width
        height = (y2-y1)* image_height

        
        result = {
            "image_id":image_id,
            "category_id": int(classId),
            "bbox":[x1,y1,width,height],
            "score": float(score)
        }
        results.append(result)
    return results


class Evaluation:
    def __init__(self,
                 path_to_images,
                 model,
                 model_name,
                 path_to_annotations,
                 input_size,
                 batch_size=32):
        self.__path_to_images = path_to_images
        self.__model = model
        self.__model_name = model_name
        self.__path_to_annotations = path_to_annotations
        self.__batch_size = batch_size
        self.__input_size = input_size

    """
        Run detection on each image an write result into Json file
    """
    def generate_results_ssd_compute_map(self):
        elapsed_time = []
        results = []
        elapsed_time = []
        
        total_image = 0
        batch_count = 0
        coco = COCO(annotation_file=self.__path_to_annotations)

        for images in load_img_from_folder(self.__path_to_images,
                                           validation_split=1,
                                           batch_size=self.__batch_size,
                                           mAP=True,
                                           input_size=self.__input_size):
            # convert images to be a tensor
            batch_count = batch_count + 1
            print(f"run evaluation for batch {batch_count}. \t")

            for item in images:                
                try:
                    input_tensort = tf.convert_to_tensor(item['np_image'])
                    input_tensort = input_tensort[tf.newaxis,...]
                    
                    start_time = time.time()
                    detections = self.__model(input_tensort)
                    end_time = time.time()

                    num_detections = int(detections.pop('num_detections'))
                    detections = {key: value[0, :num_detections].numpy()
                                for key, value in detections.items()}
                    detections['num_detections'] = num_detections 
                    # convert detection  classes to  numpy int
                    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
                    
                except :
                    continue
                if batch_count >2:
                    elapsed_time = np.append(elapsed_time, end_time - start_time)

                coco_img = coco.imgs[item['imageId']]
                img_width= coco_img['width']
                img_height = coco_img['height']
                
                boxes = detections['detection_boxes']
                classes = detections['detection_classes']
                scores = detections['detection_scores']

                result = transform_detection_to_cocoresult(image_id=item['imageId'],
                                                           image_width=img_width,
                                                           image_height=img_height,
                                                           boxes=boxes,
                                                           classes=classes,
                                                           scores=scores)
                results.extend(result)
            

            total_image =  total_image + len(images)
            if batch_count >2:
                print('average time pro batch: {:4.1f} ms'.format(elapsed_time[-len(images)] * 1000 ))
            else:
                print('Warmup...')
            print(f"Total evaluate {total_image} \t")

        ## save predicted annotation 
        print(f"save results in to json!")
        save_performance('prediction', json_data=results, file_name= self.__model_name +'.json')
        

        print(f'total time {(sum(elapsed_time)/len(elapsed_time)*1000)}')
        print('After all Evaluation FPS {:4.1f} ms '.format((sum(elapsed_time)/len(elapsed_time)*1000)))
        cocoDt = coco.loadRes(results)
        
        imgIds = sorted(coco.getImgIds())

        click.echo(click.style(f"\n compute  bbox \n", bold=True, fg='green'))
        cocoEval = COCOeval(coco, cocoDt, "bbox")
        cocoEval.params.imgIds = imgIds

        cocoEval.evaluate()        
        cocoEval.accumulate()
        cocoEval.summarize()

        print(cocoEval.stats[0])
    
    def validate_model_coco(self):
        elapsed_time = []
        results = {}
        predictions = {}
        elapsed_time = []
        
        total_image = 0
        batch_count = 0

        input_type = self.__model.inputs[0].dtype

        dataset, images_ids = load_img_from_folder_update(path_folder=self.__path_to_images,
                                                          annotations_path=self.__path_to_annotations,
                                                          batch_size=self.__batch_size,
                                                          input_size=self.__input_size,
                                                          dtype= input_type)
        for i, batch_image in enumerate(dataset):
            batch_count = batch_count + 1
            print(f"run evaluation for batch {i}. \t")
            print(len(batch_image))

            start_time = time.time()
            detections = self.__model(batch_image)
            end_time = time.time()
            print(detections)
            exit()
            for image in batch_image:
                image = image[tf.newaxis,...]
                start_time = time.time()
                detections = self.__model(image)
                end_time = time.time()
                print(detections)
            
            break
        

            
    
    def COCO_process_mAP(self,type):
        cocoGt = COCO(self.__path_to_annotations)
        cocoDt = cocoGt.loadRes(os.path.join(PATH_PERFORMANCE_INFER,self.__model_name + '.json'))
        
        imgIds = sorted(cocoGt.getImgIds())

        cocoEval = COCOeval(cocoGt, cocoDt, type)
        
        cocoEval.params.imgIds = imgIds

        cocoEval.evaluate()
        cocoEval.accumulate()

        print(cocoEval.summarize())
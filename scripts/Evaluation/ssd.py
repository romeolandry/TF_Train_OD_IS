import os
import sys
import time
import click
import random

import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import cv2 as cv

import tensorflow as tf
import numpy as np

sys.path.append(os.path.abspath(os.curdir))

from configs.run_config import *
from scripts.Evaluation.utils import *
from scripts.Evaluation.metric import *

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


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

class Inference :
    def __init__(self,
                 path_to_images,
                 path_to_labels,
                 model,
                 model_name="output",
                 threshold=0.5):
        self.__path_to_images = path_to_images
        self.__path_to_labels = path_to_labels
        self.__model = model
        self.__images_name_prefix = model_name
        self.__threshold = threshold
        self.__categories = read_label_txt(PATH_TO_LABELS_TEXT)
        ## create category index for coco 
        # self.__category_index = label_map_util.create_category_index_from_labelmap(self.__path_to_labels,use_display_name=True)

    
    '''
        draw bbox on image
    '''
    def visualize_bbox(self,image,score,bbox,classId):

        random.seed(0)

        color_str = ImageColor.getrgb(random.choice(COLOR_PANEL))

        pil_image = Image.fromarray(np.uint8(image)).convert('RGB')

        draw = ImageDraw.Draw(pil_image)
        im_width, im_height = pil_image.size

        ymin, xmin, ymax, xmax = bbox

            
        left = xmin * im_width
        right = xmax * im_width
        top = ymin * im_height
        bottom = ymax * im_height
            
        # convert to int 
        left = max(0, np.floor(left + 0.5).astype('int32'))
        right = min(im_width, np.floor(right + 0.5).astype('int32'))
        top = max(0, np.floor(top + 0.5).astype('int32'))
        bottom = min(im_height, np.floor(bottom + 0.5).astype('int32'))

        # get class text
        label = self.__categories[classId]
        # get class text
        scored_label = label + ' ' + format(score * 100, '.2f')+ '%'

        label_size = draw.textsize(scored_label)
        if top - label_size[1] >= 0:
            text_origin = tuple(np.array([left, top - label_size[1]]))
        else:
            text_origin = tuple(np.array([left, to viz_utils.visualize_boxes_and_labels_on_image_array(image_np_with_detections,
                                                                    detections['detection_boxes'][0].numpy(),
                                                                    (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
                                                                    detections['detection_scores'][0].numpy(),
                                                                    category_index,
                                                                    instance_masks=detections.get('detection_masks_reframed',None),
                                                                    use_normalized_coordinates=True,
                                                                    line_thickness=2)p + 1]))

        thickness = 4
        font = font = ImageFont.load_default()
        margin = np.ceil(0.05 * label_size[1])
        #draw.rectangle([(left, text_origin[0] - 2 * margin), (left + label_size[1],text_origin[1])],fill=color_str)
        draw.rectangle([left + thickness, top + thickness, right - thickness, bottom - thickness], outline=color_str)
        draw.text(text_origin,
                scored_label,
                font=font, 
                fill=color_str)

        np.copyto(image, np.array(pil_image))       

        return image

    ''' 
        Using SSD-savedModel to apply inference on one image
    '''
    def ssd_inference_image_cv2 (self, number_of_images=None):
    
        # read and Preprocess image 
        img = cv.imread(self.__path_to_images)
        image_np = img[:, :, [2, 1, 0]]  # BGR2RGB
       
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
                img = self.visualize_bbox(img,score,bbox,classId)

        img_path = os.path.join(PATH_DIR_IMAGE_INF,self.__images_name_prefix + "_savedmodel_ssd_cv2.png")
        cv.imwrite(img_path,img)            
        print(f"Done! image was saved into: {img_path}")
            

    def ssd_inference_webcam_saved_model(self,camera_input,camera_width,camera_height):
        extract = []        
        cap,out_file = set_input_camera(camera_input,camera_width,camera_height,self.__images_name_prefix +".mp4")        
        counter = 1
        while True:
            # Read frame from camera
            ret, image_np = cap.read()

            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            
            input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.uint8)
            detections = self.__model(input_tensor)

            ## convert all output to a numpy array
            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
                            for key, value in detections.items()}
            detections['num_detections'] = num_detections
            # detection_classes should be int64.
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)


            img = image_np.copy()

            # Extract object.
            extracted = parse_detector(image_np,
                                       detections['detection_boxes'],
                                       detections['detection_classes'],
                                       detections['detection_scores'],
                                       self.__categories,
                                       tp_th=.3,
                                       list_object_to_tracked=TRACKED_OBJECT)
            
            img_path = os.path.join(PATH_DIR_IMAGE_INF,self.__images_name_prefix + "_" + str(counter) + "_.png")
            counter = counter +1
            
            if len(extracted)> 0:
                extract.append({"image":img_path,
                                "annotation":extracted})
                cv.imwrite(img_path,img)


            # Visualize detected bounding boxes.
            for i in range(num_detections):
                classId = detections['detection_classes'][i]
                score = detections['detection_scores'][i]
                bbox = [float(v) for v in detections['detection_boxes'][i]]

                if score > self.__threshold:
                    img = self.visualize_bbox(img,score,bbox,classId)
                     # Display output
                    out_file.write(img)
                    cv.imshow(self.__images_name_prefix,img)

                    if self.__categories[classId] in TRACKED_OBJECT:
                        img_path = os.path.join(PATH_DIR_IMAGE_INF,self.__images_name_prefix + "_freezed_cv2_"+ str(i) + "_.png")
                        cv.imwrite(img_path,img)

            if cv.waitKey(25) & 0xFF == ord('q'):
                break

        cap.release()
        cv.destroyAllWindows()
        if len(extract)> 0:
            save_performance("prediction",extract, "extracted_ssd_saved.json")

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
        
        inp = img[:, :, [2, 1, 0]]  # BGR2RGB
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
                    img = self.visualize_bbox(img,score,bbox,classId)

        if not os.path.isdir(PATH_DIR_IMAGE_INF):
            os.mkdir(PATH_DIR_IMAGE_INF)
        
        img_path = os.path.join(PATH_DIR_IMAGE_INF,self.__images_name_prefix +'_ssd.png') 
        cv.imwrite(img_path,img)
        print(f"Done! image was saved into: {img_path}")
        
    
    def ssd_inference_webcam_freezed_model(self,camera_input,camera_width,camera_height):
       cap,out_file = set_input_camera(camera_input,camera_width,camera_height,self.__images_name_prefix +".mp4")
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
                img_to_infer = np.array(frame)                           
                
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
                        img = self.visualize_bbox(img_to_infer,score,bbox,classId)
                        out_file.write(img)
                        cv.imshow(self.__images_name_prefix,img)

                        if self.__categories[classId] in TRACKED_OBJECT:
                            img_path = os.path.join(PATH_DIR_IMAGE_INF,self.__images_name_prefix + "_freezed_cv2_"+ str(i) + "_.png")
                            cv.imwrite(img_path,img)
                

                if cv.waitKey(25) & 0xFF == ord('q'):
                    break
        
            cap.release()
            cv.destroyAllWindows()
        
class Evaluation:
    def __init__(self,
                 path_to_images,
                 model,
                 model_name,
                 path_to_annotations,
                 batch_size=32,
                 score_threshold=0.25,
                 iou_threshold=.5,
                 validation_split=1):
        self.__path_to_images = path_to_images
        self.__model = model
        self.__model_name = model_name
        self.__path_to_annotations = path_to_annotations
        self.__batch_size = batch_size
        self.__categories = read_label_txt(PATH_TO_LABELS_TEXT)
        self.__score_threshold = score_threshold
        self.__iou_threshold = iou_threshold
        self.__validation_split = validation_split

    """
        Run detection on each image an write result into Json file
        Return:
        results: list result in to coco formmat
        eval_imgIds: list of evaluated imageIds
        results_map: list content class_name IoU and match(True for TP and False for FP)
    """
    def generate_results_ssd_compute_map(self):
        elapsed_time = []
        results = []
        results_for_map = []
        eval_imgIds = []
        
        total_image = 0
        batch_count = 0
        cocoGt = COCO(annotation_file=self.__path_to_annotations)

        for images in load_img_from_folder(self.__path_to_images,
                                           validation_split=self.__validation_split,
                                           batch_size=self.__batch_size,
                                           mAP=True,
                                           input_size=None):
            # convert images to be a tensor
            batch_count = batch_count + 1
            print(f"\n run evaluation for batch {batch_count}\n")

            for item in images:

                coco_img = cocoGt.imgs[item['imageId']]
                img_width= coco_img['width']
                img_height = coco_img['height']
                # get Annotation Ids for the ImageId
                annotationIds = cocoGt.getAnnIds(coco_img['id'])
                # get all annatotion corresponded to this annotation Ids. get bbox segments..
                annotations = cocoGt.loadAnns(annotationIds)

                try:
                    input_tensort = tf.convert_to_tensor(item['np_image'])
                    input_tensort = input_tensort[tf.newaxis,...]
                    
                    start_time = time.time()
                    detections = self.__model(input_tensort)
                    end_time = time.time()
                except :
                    continue
                if batch_count >2:
                    elapsed_time = np.append(elapsed_time, end_time - start_time)

                num_detections = int(detections.pop('num_detections'))
                detections = {key: value[0, :num_detections].numpy()
                                for key, value in detections.items()}
                detections['num_detections'] = num_detections 
                # convert detection  classes to  numpy int
                detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
                                
                boxes = detections['detection_boxes']
                classes = detections['detection_classes']
                scores = detections['detection_scores']
                if boxes[0] is not None:
                    # for  metric computed with COCOAPi
                    result_coco_api = transform_detection_bbox_to_cocoresult(image_id=item['imageId'],
                                                            image_width=img_width,
                                                            image_height=img_height,
                                                            boxes=boxes,
                                                            classes=classes,
                                                            scores=scores)
                    results.extend(result_coco_api)
                    eval_imgIds.append(item['imageId'])

                    # for metric computed without COCOAPi
                    result_simple = compute_iou_of_prediction_bbox(image_width=img_width,
                                                                   image_height=img_height,
                                                                   boxes=boxes,
                                                                   classes= classes,
                                                                   scores=scores,
                                                                   coco_annatotions= annotations,
                                                                   score_threshold = self.__score_threshold,
                                                                   iou_threshold =self.__iou_threshold,
                                                                   categories = self.__categories)

                    results_for_map.extend(result_simple)
            

            total_image =  total_image + len(images)
            if batch_count >2:
                print('time pro batch: {:4.1f} s'.format((sum(elapsed_time[-self.__batch_size:]))))
            else:
                print('Warmup...')
            print(f"Total evaluate {total_image}")
        
        print(f'total time sum {sum(elapsed_time)}')
        print(f'total time len {len(elapsed_time)}')
        print('After all Evaluation FPS {:4.1f} first methode '.format((total_image/sum(elapsed_time))))
        print('After all Evaluation FPS {:4.1f} second methode '.format(1000/((sum(elapsed_time)/len(elapsed_time))*1000)))
        
        return results,eval_imgIds, results_for_map
    
    def COCO_process_mAP(self, results, evaluated_imageIds):
        print("*"*50)
        print("Compute metric with COCOApi")
        print("*"*50)
        cocoGt = COCO(self.__path_to_annotations)
        cocoDt = cocoGt.loadRes(results)

        cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
        
        cocoEval.params.imgIds = sorted(evaluated_imageIds) 

        cocoEval.evaluate()        
        cocoEval.accumulate()
        cocoEval.summarize()

        print(cocoEval.stats[0])

    def mAP_without_COCO_API(self,results, per_class):

        print("*"*50)
        print(f"Compute metric without COCOApi per class : {per_class}")
        print("*"*50)
        # computer mAP
        ap_dictionary = get_map(results, per_class=per_class)
        print(f"{ap_dictionary}")
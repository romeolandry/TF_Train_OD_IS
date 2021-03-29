import os
import sys
import time
import click
import random
import cv2 as cv
import tensorflow as tf
import numpy as np
import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont

sys.path.append(os.path.abspath(os.curdir))

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

from configs.run_config import *
from scripts.Evaluation.metric import *
from scripts.Evaluation.utils import *



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


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
                 threshold=0.5):
        self.__path_to_images = path_to_images
        self.__path_to_labels = path_to_labels
        self.__model = model
        self.__images_name_prefix = model_name
        self.__threshold = threshold
        self.__categories = read_label_txt(PATH_TO_LABELS_TEXT)
    
    
    
    
    '''
        draw bbox and mask on image
    '''
    def visualize_bbox_mask_pil(self,image,score,bbox,mask,classId):

        random.seed(0)

        classId = classId.astype(np.int64)

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
            text_origin = tuple(np.array([left, top + 1]))

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
        ## Mask
        color_str = ImageColor.getrgb(random.choice(COLOR_PANEL))
        pil_image_mask = Image.fromarray(image)
             
        solid_color = np.expand_dims(np.ones_like(mask), axis=2) * np.reshape(list(color_str), [1, 1, 3])
        pil_solid_color = Image.fromarray(np.uint8(solid_color)).convert('RGBA')
        pil_mask = Image.fromarray(np.uint8(255.0* 0.4*(mask > 0))).convert('L')
        pil_image_mask = Image.composite(pil_solid_color, pil_image_mask, pil_mask)

        np.copyto(image, np.array(pil_image_mask))

        return image
    
    '''
        Using Mask SavedModel to apply mask on a given image
        this  function required Tensorflow Object detection API to apply mask
        else its will just draw boxes on frame
    '''
    def mask_inference_image_cv(self):
        
        # read and Preprocess image 
        img = cv.imread(self.__path_to_images)

        image_np = img[:, :, [2, 1, 0]]  # BGR2RGB
        # convert images to be a tensor
        input_tensor = tf.convert_to_tensor(image_np)
        input_tensor = input_tensor[tf.newaxis, ...]

        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.uint8)
        # Apply inference 
        detections = self.__model(input_tensor)

        image_np_with_detections = image_np.copy()

        try:
            from object_detection.utils import ops as utils_ops
            from object_detection.utils import label_map_util
            from object_detection.utils import visualization_utils as viz_utils
            
            category_index = label_map_util.create_category_index_from_labelmap(self.__path_to_labels,use_display_name=True)  
            
            label_id_offset=0

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
                                                                category_index,
                                                                instance_masks=detections.get('detection_masks_reframed',None),
                                                                use_normalized_coordinates=True,
                                                                max_boxes_to_draw = 100,
                                                                min_score_thresh = self.__threshold,
                                                                agnostic_mode = False,
                                                                line_thickness=2)
        except ImportError:
            click.echo(click.style(f"\n The mask wil not be apply. Object detection API is not availabe \n", bold=True, fg='red'))
            
            classIds = detections['detection_classes'][0].numpy()
            scores = detections['detection_scores'][0].numpy()
            boxes = detections['detection_boxes'][0].numpy()
            masks = detections['detection_masks'][0].numpy()
            
            # Visualize detected bounding boxes.   
            for i in range(boxes.shape[0]):
                if scores[i] > self.__threshold:
                    score = scores[i]
                    bbox = boxes[i]
                    instance_mask = masks[i]
                    classId = classIds[i]                

                    img = self.visualize_bbox_mask_pil(image_np_with_detections,score,bbox,instance_mask,classId)

        img_path = os.path.join(PATH_DIR_IMAGE_INF,self.__images_name_prefix + "_savedmodel_.png")
        cv.imwrite(img_path,image_np_with_detections)
            
        cv.imshow('TensorFlow Mask-ResNet_new', image_np_with_detections)
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

                print(f" This model compute {int(cap.get(cv.CAP_PROP_FPS))} FPS ")
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
        
        # image_np = cv.resize(img, (self.__model_image_size[0], self.__model_image_size[1]))
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
            height,width = image_np.shape[:2]
            detection_masks_reframed = reframe_box_masks_to_image_masks(detection_masks, detection_boxes,height, width)
            detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,tf.uint8)
            detections['detection_masks_reframed'] = detection_masks_reframed.numpy()
            
        label_id_offset=0


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
    
    

    '''
        Using Mask SavedModel to apply mask captured image
        this  function required Tensorflow Object detection API to apply mask
        else its will just draw boxes on frame
    '''
    def mask_inference_webcam_2(self, camera_input, camera_width, camera_height):
       
        cap,out_file = set_input_camera(camera_input,camera_width,camera_height,self.__images_name_prefix +".mp4")        
        counter = 1
        while True:
            # Read frame from camera
            ret,image_np = cap.read()
            # expand image to have shape :[1, None, None,3]
            image_np_expanded = np.expand_dims(image_np, axis=0)

            input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0),dtype=tf.uint8)
            detections = self.__model(input_tensor)

            image_np_with_detections = image_np.copy()

            try:
                from object_detection.utils import ops as utils_ops
                from object_detection.utils import label_map_util
                from object_detection.utils import visualization_utils as viz_utils
                
                category_index = label_map_util.create_category_index_from_labelmap(self.__path_to_labels,use_display_name=True)  
                
                label_id_offset=0

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
                                                                    category_index,
                                                                    instance_masks=detections.get('detection_masks_reframed',None),
                                                                    use_normalized_coordinates=True,
                                                                    max_boxes_to_draw = 100,
                                                                    min_score_thresh = self.__threshold,
                                                                    agnostic_mode = False,
                                                                    line_thickness=2)
            except ImportError:
                click.echo(click.style(f"\n The mask wil not be apply. Object detection API is not availabe \n", bold=True, fg='red'))
            
                classIds = detections['detection_classes'][0].numpy()
                scores = detections['detection_scores'][0].numpy()
                boxes = detections['detection_boxes'][0].numpy()
                masks = detections['detection_masks'][0].numpy()
            
                # Visualize detected bounding boxes.   
                for i in range(boxes.shape[0]):
                    if scores[i] > self.__threshold:
                        score = scores[i]
                        bbox = boxes[i]
                        instance_mask = masks[i]
                        classId = classIds[i]         

                        img = self.visualize_bbox_mask_pil(image_np_with_detections,score,bbox,instance_mask,classId)

            img_path = os.path.join(PATH_DIR_IMAGE_INF,self.__images_name_prefix + "_savedmodel_.png")
            out_file.write(image_np_with_detections)
            
            
            cv.imshow(self.__images_name_prefix, image_np_with_detections)
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
                 batch_size,
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
    def generate_results_mask_compute_map(self):
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
            print(f"run evaluation for batch {batch_count} \t")
            for item in images:               
                
                coco_img = cocoGt.imgs[item['imageId']]
                img_width= coco_img['width']
                img_height = coco_img['height']
                # get Annotation Ids for the ImageId
                annotationIds = cocoGt.getAnnIds(coco_img['id'])
                # get all annatotion corresponded to this annotation Ids. get bbox segments..
                annotations = cocoGt.loadAnns(annotationIds)
                
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
                if batch_count >2:
                    elapsed_time = np.append(elapsed_time, end_time - start_time)
               

                boxes = detections['detection_boxes'][0]
                classes = detections['detection_classes'][0]
                scores = detections['detection_scores'][0]
                masks  = detections['detection_masks'][0]

                if boxes[0] is not None:
                    result_coco_api = transform_detection_mask_to_cocoresult(image_id= item['imageId'],
                                                                    image_width=img_width,
                                                                    image_height=img_height,
                                                                    boxes=boxes,
                                                                    classes=classes,
                                                                    masks=masks,
                                                                    scores=scores)
                                
                    results.extend(result_coco_api)
                    eval_imgIds.append(item['imageId'])

                    # for metric computed without COCOAPi
                    result_simple = compute_iou_of_prediction_bbox_segm(image_width=img_width,
                                                                    image_height=img_height,
                                                                    boxes=boxes,
                                                                    classes= classes,
                                                                    scores=scores,
                                                                    masks=masks,
                                                                    coco_annatotions= annotations,
                                                                    score_threshold = self.__score_threshold,
                                                                    iou_threshold =self.__iou_threshold,
                                                                    categories = self.__categories)

                    results_for_map.extend(result_simple)
                               
            total_image = total_image + len(images)

            if batch_count >2:
                print('average time pro batch: {:4.1f}s'.format(sum(elapsed_time[-self.__batch_size:])))
            else:
                print('Warmup...')

            print(f"Total evaluate {total_image} \t")        
               
        print('After all Evaluation FPS {:4.1f} first methode '.format((total_image/sum(elapsed_time))))
        print('After all Evaluation FPS {:4.1f} second methode '.format(1000/((sum(elapsed_time)/len(elapsed_time))*1000)))
        
        return results,eval_imgIds, results_for_map
            

    def COCO_process_mAP(self, results, evaluated_imageIds):
        print("*"*50)
        print("Compute metric bbox with COCOApi")
        print("*"*50)

        click.echo(click.style(f"\n compute  bbox \n", bold=True, fg='green'))
        cocoGt = COCO(self.__path_to_annotations)
        cocoDt = cocoGt.loadRes(results)
        
        cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
        cocoEval.params.imgIds = evaluated_imageIds        
        
        cocoEval.evaluate()        
        cocoEval.accumulate()
        cocoEval.summarize()

        print(cocoEval.stats[0])

        print("*"*50)
        click.echo(click.style(f"\n compute  segmentation \n", bold=True, fg='green'))
        print("*"*50)

        cocoEval = COCOeval(cocoGt, cocoDt, "segm")
        cocoEval.params.imgIds = evaluated_imageIds        
        
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
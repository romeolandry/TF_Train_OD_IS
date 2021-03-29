import os
import sys
import time
import click
import random
import itertools
import colorsys
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from skimage.measure import find_contours
import cv2 as cv


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import tensorflow as tf
import numpy as np

sys.path.append(os.path.abspath(os.curdir))

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

from configs.run_config import *
from scripts.Evaluation.metric import *
from scripts.Evaluation.utils import *

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils


import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont


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

def class_color(id,prob, hsv):
    _hsv = list(hsv[id])
    # _hsv[2]=random.uniform(0.8, 1)
    _hsv[2]=prob
    color = colorsys.hsv_to_rgb(*_hsv)
    return color

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
    
    
    def reframe_box_masks_to_image_masks(self,box_masks, boxes, image_height,
                                     image_width, resize_method='bilinear'):
        """Transforms the box masks back to full image masks.

        Embeds masks in bounding boxes of larger masks whose shapes correspond to
        image shape.

        Args:
            box_masks: A tensor of size [num_masks, mask_height, mask_width].
            boxes: A tf.float32 tensor of size [num_masks, 4] containing the box
                corners. Row i contains [ymin, xmin, ymax, xmax] of the box
                corresponding to mask i. Note that the box corners are in
                normalized coordinates.
            image_height: Image height. The output mask will have the same height as
                        the image height.
            image_width: Image width. The output mask will have the same width as the
                        image width.
            resize_method: The resize method, either 'bilinear' or 'nearest'. Note that
            'bilinear' is only respected if box_masks is a float.

        Returns:
            A tensor of size [num_masks, image_height, image_width] with the same dtype
            as `box_masks`.
        """
        resize_method = 'nearest' if box_masks.dtype == tf.uint8 else resize_method
        # TODO(rathodv): Make this a public function.
        def reframe_box_masks_to_image_masks_default():
            """The default function when there are more than 0 box masks."""
            def transform_boxes_relative_to_boxes(boxes, reference_boxes):
                boxes = tf.reshape(boxes, [-1, 2, 2])
                min_corner = tf.expand_dims(reference_boxes[:, 0:2], 1)
                max_corner = tf.expand_dims(reference_boxes[:, 2:4], 1)
                denom = max_corner - min_corner
                # Prevent a divide by zero.
                denom = tf.math.maximum(denom, 1e-4)
                transformed_boxes = (boxes - min_corner) / denom
                return tf.reshape(transformed_boxes, [-1, 4])

            box_masks_expanded = tf.expand_dims(box_masks, axis=3)
            num_boxes = tf.shape(box_masks_expanded)[0]
            unit_boxes = tf.concat(
                [tf.zeros([num_boxes, 2]), tf.ones([num_boxes, 2])], axis=1)
            reverse_boxes = transform_boxes_relative_to_boxes(unit_boxes, boxes)
            
            resized_crops = tf.image.crop_and_resize(
                image=box_masks_expanded,
                boxes=reverse_boxes,
                box_ind=tf.range(num_boxes),
                crop_size=[image_height, image_width],
                method=resize_method,
                extrapolation_value=0)
            return tf.cast(resized_crops, box_masks.dtype)

        image_masks = tf.cond(
            tf.shape(box_masks)[0] > 0,
            reframe_box_masks_to_image_masks_default,
            lambda: tf.zeros([0, image_height, image_width, 1], box_masks.dtype))
        return tf.squeeze(image_masks, axis=3)
    
    '''
        draw bbox and mask on image
    '''
    def visualize_bbox_mask(self,image,score,bbox,mask,classId):

        random.seed(0)
        N=90
        brightness = 1.0
        hsv = [(i / N, 1, brightness) for i in range(N)]
        random.shuffle(hsv)
        classId = classId.astype(np.int64)

        # get random color
        #color = map(lambda c: colorsys.hsv_to_rgb(*c), hsv)
        #color_box = map(lambda c: colorsys.hsv_to_rgb(*c), hsv)

        color = class_color(classId,score*score*score*score,hsv)
        
        
        height,width = image.shape[:2]

        img = image.copy()

        categories = read_label_txt(PATH_TO_LABELS_TEXT)


        bbox = bbox * np.array([height, width, height, width])
        startX, startY, endX, endY = bbox.astype("int")
        font = cv.FONT_HERSHEY_COMPLEX      

        cv.rectangle(img, (startY, startX), (int(endY), int(endX)), [int(x*255) for x in (color)], thickness=2)

        # get class text
        label = categories[classId]
        # get class text
        scored_label = label + ' ' + format(score * 100, '.2f')+ '%'

        cv.putText(img, scored_label, (int(startY)+10, int(startX)+20),font, 1, [int(x*255) for x in (color)], thickness=1)

        # Mask
        # Apply mask  on image
        boxW = endX - startX
        boxH = endY - startY

        # extract the ROI of the image
        #roi = img[startY:endY, startX:endX]

        # mask = cv.resize(mask, (width, height ),interpolation=cv.INTER_NEAREST)
        for c in range(3):
            img[:,:,c] = np.where(mask ==1,
                                  image[:,:,c]*
                                  (1-0.5)+ 0.5 * color[c]*255,
                                  img[:,:,c])
        
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
      
        #contours, hierarchy = cv.findContours(padded_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        #cv.drawContours(img, contours, -1, (0,255,0), 1)
        # cv.polylines(mask, contours, -1, [int(x*255) for x in (color)],4)
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            # print(verts.shape)
            Polygon(verts, facecolor="none", edgecolor=color)
            # ax.add_patch(p)
            pts = np.array(verts.tolist(), np.int32)
            pts = pts.reshape((-1,1,2))
            cv.polylines(img,[pts],True,[int(x*255) for x in (color)],4)

        return img

    '''
        draw bbox and mask on image
    '''
    def visualize_bbox_mask_pil(self,image,score,bbox,mask,classId):

        random.seed(0)

        classId = classId.astype(np.int64)

        categories = read_label_txt(PATH_TO_LABELS_TEXT)

        color_str = ImageColor.getrgb(random.choice(COLOR_PANEL))
                
        pil_image = Image.fromarray(np.uint8(image)).convert('RGB')

        draw = ImageDraw.Draw(pil_image)
        im_width, im_height = pil_image.size

        ymin, xmin, ymax, xmax = bbox

        (left, right, top, bottom) = (xmin * im_width,
                                    xmax * im_width,
                                    ymin * im_height,
                                    ymax * im_height
                                    )
            
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
        label = categories[classId]
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
                fill=ImageColor.getrgb("black"))

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
        draw bbox and mask on image
    '''
    def visualize_bbox_mask_cv2(self,image,score,bbox,mask,classId):
        
        height,width = image.shape[:2]

        img = image.copy()

        categories = read_label_txt(PATH_TO_LABELS_TEXT)

        # get class text
        label = categories[classId]
        # get class text
        scored_label = label + ' ' + format(score * 100, '.2f')+ '%'

        bbox = bbox * np.array([height, width, height, width])
        startX, startY, endX, endY = bbox.astype("int")
        font = cv.FONT_HERSHEY_COMPLEX

        # mask
        boxW = endX - startX
        boxH = endY - endX
        mask = cv.resize(mask,(boxW,boxH),interpolation=cv.INTER_NEAREST)

        # mask_cls = (mask > .3)
        # extract ROI of image
        roi =  img[startY:endY +1, startX:endX + 1]

        blended = (((255,0,255)) + (roi)).astype("uint8")
        #img[startY:endY +1, startX:endX+1][mask] = blended

        font = cv.FONT_HERSHEY_COMPLEX

        cv.rectangle(img, (startY, startX), (int(endY), int(endX)), (125, 255, 51), thickness=2)
        cv.putText(img, scored_label, (int(startY)+10, int(startX)+20),font, 1, (0, 255, 0), thickness=1)

        return img

    
    
    def mask_inference_image_cv(self):
        
        # read and Preprocess image 
        img = cv.imread(self.__path_to_images)
        
        #image_np = cv.resize(img, (self.__model_image_size[0], self.__model_image_size[1]))
        image_np = img[:, :, [2, 1, 0]]  # BGR2RGB
        # read frozen Graph       
        elapsed_time = 0

        # convert images to be a tensor
        input_tensor = tf.convert_to_tensor(image_np)
        input_tensor = input_tensor[tf.newaxis, ...]

        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.uint8)
        # Apply inference 
        detections = self.__model(input_tensor)

        detection_masks = tf.convert_to_tensor(detections['detection_masks'][0])
        detection_boxes = tf.convert_to_tensor(detections['detection_boxes'][0])

        # Reframe the the bbox mask to the image size.

        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks, detection_boxes,image_np.shape[0], image_np.shape[1])
        detectihon_masks_reframed = tf.cast(detection_masks_reframed > 0.3,tf.uint8)
        detections['detection_masks_reframed'] = detection_masks_reframed.numpy()

        classIds = detections['detection_classes'][0].numpy()
        scores = detections['detection_scores'][0].numpy()
        boxes = detections['detection_boxes'][0].numpy()
        masks = detections['detection_masks'][0].numpy()
        instance_masks = detections.get('detection_masks_reframed',None)
        
        # Visualize detected bounding boxes.   
        for i in range(boxes.shape[0]):
            if scores[i] > self.__threshold:
                score = scores[i]
                bbox = boxes[i]
                instance_mask = instance_masks[i]
                classId = classIds[i]
                

                img = self.visualize_bbox_mask_pil(img,score,bbox,instance_mask,classId)

        img_path = os.path.join(PATH_DIR_IMAGE_INF,self.__images_name_prefix + "_savedmodel_new_cv2.png")
        cv.imwrite(img_path,img)
            
        cv.imshow('TensorFlow SSD-ResNet_new', img)
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
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks, detection_boxes,height, width)
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

        cap = set_input_camera(camera_input,camera_width,camera_height)

        while True:
            # Read frame from camera
            ret, image_np = cap.read()

            # expand image to have shape :[1, None, None,3]
            image_np_expanded = np.expand_dims(image_np, axis=0)

            input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0),dtype=tf.float32)
            detections = self.__model(input_tensor)

            label_id_offset = 0
            image_np_with_detections = image_np.copy()
            
            if 'detection_masks' in detections:
    
                detection_masks = tf.convert_to_tensor(detections['detection_masks'][0])
                detection_boxes = tf.convert_to_tensor(detections['detection_boxes'][0])

                # Reframe the the bbox mask to the image size.

                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks, detection_boxes,image_np.shape[0], image_np.shape[1])
                detectihon_masks_reframed = tf.cast(detection_masks_reframed > 0.5,tf.uint8)
                detections['detection_masks_reframed'] = detection_masks_reframed.numpy()
            
            viz_utils.visualize_boxes_and_labels_on_image_array(image_np_with_detections,
                                                detections['detection_boxes'][0].numpy(),
                                                (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
                                                detections['detection_scores'][0].numpy(),
                                                self.__category_index,
                                                instance_masks=detections.get('detection_masks_reframed',None),
                                                use_normalized_coordinates=True,
                                                max_boxes_to_draw= 200,
                                                min_score_thresh=.80,
                                                agnostic_mode=False,
                                                line_thickness=8)


            # Display output
            cv.imshow('Instance Segmentation', cv.resize(image_np_with_detections, (800, 600)))

            if cv.waitKey(25) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv.destroyAllWindows()

    '''
        Using SSD-resnet50 v2  to apply inference on a list of image
        The image will be progressively load and inference
        the output images will be save in to 'images_inferences'
    '''
    def mask_inference_webcam_2(self, camera_input, camera_width, camera_height):

        cap = set_input_camera(camera_input,camera_width,camera_height)
        out_file = cv.VideoWriter(self.__images_name_prefix +".mp4",cv.VideoWriter_fourcc('M','J','P','G'), 10, (int(cap.get(3)),int(cap.get(4))))
        counter = 1
        while True:
            # Read frame from camera
            ret, image_np = cap.read()

            # expand image to have shape :[1, None, None,3]
            image_np_expanded = np.expand_dims(image_np, axis=0)

            input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0),dtype=tf.uint8)
            detections = self.__model(input_tensor)

            label_id_offset = 0
            image_np_with_detections = image_np.copy()
            
            if 'detection_masks' in detections:
    
                detection_masks = tf.convert_to_tensor(detections['detection_masks'][0])
                detection_boxes = tf.convert_to_tensor(detections['detection_boxes'][0])

                # Reframe the the bbox mask to the image size.

                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks, detection_boxes,image_np.shape[0], image_np.shape[1])
                detectihon_masks_reframed = tf.cast(detection_masks_reframed > 0.5,tf.uint8)
                detections['detection_masks_reframed'] = detection_masks_reframed.numpy()

            viz_utils.visualize_boxes_and_labels_on_image_array(image_np_with_detections,
                                                detections['detection_boxes'][0].numpy(),
                                                (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
                                                detections['detection_scores'][0].numpy(),
                                                self.__category_index,
                                                instance_masks=detections.get('detection_masks_reframed',None),
                                                use_normalized_coordinates=True,
                                                max_boxes_to_draw= 200,
                                                min_score_thresh=.80,
                                                agnostic_mode=False,
                                                line_thickness=8)


            # Display output
            cv.imshow('Instance Segmentation', cv.resize(image_np_with_detections, (800, 600)))

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
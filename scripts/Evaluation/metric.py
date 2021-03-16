import os
import sys
sys.path.append(os.path.abspath(os.curdir))

from configs.run_config import *
import numpy as np
import cv2 as cv
from pycocotools import mask as maskUtils

def voc_ap(rec=[], pre=[]):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0,0.0)
    rec.append(1.0)
    mrec = rec[:]

    pre.insert(0,0.0)
    pre.append(0.0)
    mpre = pre[:]

    """
     This part makes the precision monotonically decreasing
        (goes from the end to the beginning)
        matlab: for i=numel(mpre)-1:-1:1
                    mpre(i)=max(mpre(i),mpre(i+1));
    """

    for i in range(len(mpre)-2,-1,-1):
        mpre[i]=max(mpre[i],mpre[i+1])

    """ Get a list of indexes where the recall changes
        matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    """
    i_list = []
    for i in range(1,len(mrec)):
        i_list.append(i)

    """
     The Average Precision (AP) is the area under the curve
        (numerical integration)
        matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
    return ap, mrec, mpre


'''
    Compute IoU of bbox
    args:
    ground_truth: real bbox from coco with format (x1, y1, x2, y2) (upperleft,bottonright)
    predicted_box: predicted bbox with format (x1, y1, x2, y2) (upperleft,bottonright)

    Return: IoU of the both bbox
'''
def iou_bbox(ground_truth, predicted_box, smooth=1e-5):
    
    # intersection rectangle coordinate
    x_inter_a = max(ground_truth[0],predicted_box[0])
    y_inter_a = max(ground_truth[1],predicted_box[1])
    x_inter_b = min(ground_truth[2],predicted_box[2])
    y_inter_b = min(ground_truth[3],predicted_box[3])

    # area of intersection rectangle
    #intersection_area = abs(max(( x_inter_b - x_inter_a),0) * max ((y_inter_b + y_inter_a),0))

    width = x_inter_b - x_inter_a
    height = y_inter_b - y_inter_a

    # compute the area of intersection rectangle
    intersection_area = abs(max((x_inter_b - x_inter_a, 0)) * max((y_inter_b - y_inter_a), 0))
    if intersection_area == 0:
        return 0.0
    # Area of both box
    ground_truth_area = abs((ground_truth[2] - ground_truth[0]) * (ground_truth[3] - ground_truth[1]))
    predicted_box_area = abs((predicted_box[2] - predicted_box[0]) * (predicted_box[3] - predicted_box[1]))

    area_combined = ground_truth_area + predicted_box_area - intersection_area

    # Compute IoU
    iou = intersection_area/area_combined+smooth

    return iou

def iou_mask(ground_truth_mask, predicted_mask, smooth=1e-5):
    pass


'''
    Transform box(upperleft,bottomright ) to (upperleft,width,height)
    args:
    image_id: coco image id
    image_width : image width
    image_height image height
    boxes: list predicted bbox
    classes: list predicted class
    scores: list predicted score

    Return: dictionary result in coco format
'''
def transform_detection_bbox_to_cocoresult(image_id,
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
        x1 = float(x1 * image_width) #* image_width
        y1 = float(y1   * image_height) #  * image_height
        width = float((x2-x1)  * image_width) # * image_width
        height = float((y2-y1) * image_height) # * image_height

        
        result = {
            "image_id":image_id,
            "category_id": int(classId),
            "bbox":[y1,x1,width,height],
            "score": float(score)
        }
        results.append(result)
    return results



'''
    Transform box(upperleft,bottomright ) to (upperleft,width,height)
    args:
    image_id: coco image id
    image_width : image width
    image_height image height
    boxes: list predicted bbox
    classes: list predicted class
    scores: list predicted score
    masks: list predicted mask
    Return: dictionary result in coco format
'''
def transform_detection_mask_to_cocoresult(image_id,
                                           image_width,
                                           image_height,
                                           boxes,
                                           masks,
                                           classes,
                                           scores):
    
    boxes = boxes.numpy()
    masks = masks.numpy()
    scores = scores.numpy()
    classes = classes.numpy()
    
    assert boxes.shape[0] == masks.shape[0] == classes.shape[0] == scores.shape[0]

    if boxes is None:
        return []

    results = []

    
    for i in range(boxes.shape[0]):
        classId = classes[i]
        score = scores[i]
        bbox = boxes[i]
        y1,x1,y2,x2 = list(bbox)
        # normalize 
        x1 = float(x1 * image_width)
        y1 = float(y1 * image_height)
        width = float((x2-x1)* image_width)
        height = float((y2-y1)*image_height)

        mask = cv.resize(masks[i], (image_width,image_height), interpolation=cv.INTER_NEAREST)
        mask = np.uint8(masks[i])
       
        result = {
            'image_id':image_id,
            'category_id': int(classId),
            'bbox':[x1,y1,width,height],
            'score': float(score),
            'segmentation': maskUtils.encode(np.asfortranarray(mask))
        }
        results.append(result)
    return results


'''
    compute IoU of one Prediction 
    args:
    class_name: coco image id
    image_width : image width
    image_height image height
    boxes: list predicted bbox
    classes: list predicted class
    scores: list predicted score
    coco_annatotions: coco annotation for the prediction
    score_threshold:
    iou_threshold: to define true_position(match=true) and false_positive(match=false)

    Return: dictionary result{class_name: iuo: match:}
'''

def compute_iou_of_prediction_bbox(image_width,
                                   image_height,
                                   boxes,
                                   classes,
                                   scores,
                                   coco_annatotions,
                                   score_threshold,
                                   iou_threshold,
                                   categories):
    assert boxes.shape[0] == classes.shape[0] == scores.shape[0]

    results = []

    for i in range(boxes.shape[0]):
        classId = classes[i]
        score = round(scores[i],4)
        pred_bbox = np.around(boxes[i],1)
        pred_bbox =  pred_bbox * np.array([image_width,image_height,image_width,image_height])

        if score < score_threshold:
            continue

        for annotation in coco_annatotions:
            x,y,w,h = annotation['bbox']
            ground_truth =[x,y, x+w, y+h]

            iou = iou_bbox(ground_truth,pred_bbox)
            match = False

            if (iou >= iou_threshold):
                if not match:
                    match = True
                else:
                    match = False
            else:
                match = False
            results.append({
                "class_name":  categories[classId],
                "iou":iou,
                "match": match
            })

    return results

'''
    compute IoU of one Prediction 
    args:
    class_name: coco image id
    image_width : image width
    image_height image height
    boxes: list predicted bbox
    classes: list predicted class
    scores: list predicted score
    coco_annatotions: coco annotation for the prediction
    score_threshold:
    iou_threshold: to define true_position(match=true) and false_positive(match=false)

    Return: dictionary result{class_name: iuo: match:}
'''

def compute_iou_of_prediction_bbox_segm(image_width,
                                        image_height,
                                        boxes,
                                        classes,
                                        scores,
                                        masks,
                                        coco_annatotions,
                                        score_threshold,
                                        iou_threshold,
                                       categories):
    
    boxes = boxes.numpy()
    masks = masks.numpy()
    scores = scores.numpy()
    classes = classes.numpy()

    assert boxes.shape[0] == classes.shape[0] == scores.shape[0] == masks.shape[0]

    results_box = []
    results_mask = []

    for i in range(boxes.shape[0]):
        classId = classes[i]
        score = round(scores[i],4)
        pred_bbox = boxes[i]
        pred_bbox =  pred_bbox * np.array([image_width,image_height,image_width,image_height])

        if score < score_threshold:
            continue

        for annotation in coco_annatotions:
            x,y,w,h = annotation['bbox']
            ground_truth =[x,y, x+w, y+h]

            iou = iou_bbox(ground_truth,pred_bbox)
            match = False

            if (iou >= iou_threshold):
                if not match:
                    match = True
                else:
                    match = False
            else:
                match = False
            results_box.append({
                "class_name":  categories[classId],
                "iou":iou,
                "match": match
            })
    
    return results_box

def comput_iou_mask(mask_pred,
                    mask_ground_truth):
    assert mask_ground_truth.shape == mask_pred.shape, 'Input masks should be same shape, instead are {}, {}'.format(mask_ground_truth.shape, mask_pred.shape)
    
    # intersection and union shapes are batch_size * n_classes (values = area in pixels)
    axes = (1,2) # W,H axes of each image
    intersection = np.sum(np.abs(mask_pred * mask_ground_truth), axis=axes) # or, np.logical_and(y_pred, y_true) for one-hot
    mask_sum = np.sum(np.abs(mask_ground_truth), axis=axes) + np.sum(np.abs(mask_pred), axis=axes)
    union = mask_sum  - intersection # or, np.logical_or(y_pred, y_true) for one-hot

    smooth = .001
    iou = (intersection + smooth) / (union + smooth)

def get_map(results, per_class):
    # count class_name in to result
    counter_per_class = {}
    sum_AP = 0.0
    ap_dictionary = {}

    for result in results:
        if result['class_name'] in counter_per_class:            
            counter_per_class[result['class_name']] +=1
        else:
            counter_per_class[result['class_name']] =1
        
    
    list_class_name = list(counter_per_class.keys())
    n_classes = len(list_class_name) # number of availabe class
    if not per_class:
        tp = [0] * len(results)
        fp = [0] * len(results)
        print(f"Computer Average Precision over all Class")
        for idx, result in enumerate(results):
            
            if result['match']:
                tp[idx] = 1
            else:
                fp[idx] = 1
            
        # compute precision/recall
        cumsum = 0

        for idx, val in enumerate(fp):
            fp[idx] += cumsum
            cumsum += val
        cumsum = 0
        for idx, val in enumerate(tp):
            tp[idx] += cumsum
            cumsum += val

        rec = tp[:]
        for idx, val in enumerate(tp):
            rec[idx] = float(tp[idx]) /n_classes
        
        #print(rec)
        prec = tp[:]
        for idx, val in enumerate(tp):
            prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])

        ap,mrec,mprec = voc_ap(rec,prec)
        sum_AP += ap
        ap_dictionary["ap"] = ap*100
    else:
        print(f"Computer Average Precision for each Class")
        result_per_class = []
        for class_name in list_class_name:
            
            for idx, result in enumerate(results):
                if result['class_name'] == class_name:
                    result_per_class.append(results[idx])

            tp = [0] * len(result_per_class)
            fp = [0] * len(result_per_class)

            for idx, result in enumerate(result_per_class):
                if result['match']:
                    tp[idx] = 1
                else:
                    fp[idx] = 1
            
            # compute precision/recall
            cumsum = 0

            for idx, val in enumerate(fp):
                fp[idx] += cumsum
                cumsum += val
            cumsum = 0
            for idx, val in enumerate(tp):
                tp[idx] += cumsum
                cumsum += val

            rec = tp[:]
            for idx, val in enumerate(tp):
                rec[idx] = float(tp[idx]) / counter_per_class[class_name]
            
            #print(rec)
            prec = tp[:]
            for idx, val in enumerate(tp):
                prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])

            ap,mrec,mprec = voc_ap(rec,prec)
            sum_AP += ap
            ap_dictionary[class_name] = ap*100
    
    print(f"number of nb:  {n_classes} ")
    mAP = sum_AP/n_classes
    ap_dictionary['mAP'] = mAP * 100

    return ap_dictionary
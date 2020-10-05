"""TensorFlow provide an list of pre-trained Model available on 
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
for this project we choose "SSD MobileNet V1 FPN 640x640" for Object detectiopn and "Mask R-CNN Inception ResNet V2 1024x1024"
"""

## DataSet
COCO_YEARS= 2017
PATH_IMAGES = "images"
PATH_ANNOTATIONS="annotations"

LIST_MODEL_TO_DOWNLOAD = ["ssd_mobilenetv1", "maskrcnn"]
SSD_MODEL_URL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8.tar.gz"

MASK_RCNN_URL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8.tar.gz"
PRE_TRAINED_MODEL_DIR_PATH = "pre_trained_models"
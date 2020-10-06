"""TensorFlow provide an list of pre-trained Model available on 
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
for this project we choose "SSD MobileNet V1 FPN 640x640" for Object detectiopn and "Mask R-CNN Inception ResNet V2 1024x1024"
"""

## DataSet
COCO_YEARS= 2017
PATH_IMAGES = "images"
PATH_ANNOTATIONS="annotations"
PATH_TRAINED_MODELS = "models"
PREFIX_MODEL_NAME = "my_"
PATH_TO_EXPORT_DIR = "exported_models"
SUFIX_EXPORT = "_saved"

LIST_MODEL_TO_DOWNLOAD = {
    "ssd_resnet50_v1":"http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz",
    "ssd_mobilenetv1":"http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8.tar.gz",
    "maskrcnn":"http://download.tensorflow.org/models/object_detection/tf2/20200711/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8.tar.gz"
}

PATH_PRE_TRAINED_MODELS = "pre_trained_models"

## Train Configurations 

NUM_TRAIN_STEP = 1000
EVAL_ON_TRAIN_DATA = False # only supported in distributed training
SAMPLE_OF_N_EVAL = None
SAMPLE_OF_N_EVAL_ON_TRAIN = None # used when Eval on train data is true
CHECKPOINT_EVERY_N_STEP = 1000
RECORD_SUMMARY = True

# Evalution 
CHECKPOINT_DIR = None # path to checkpoint of model to evaluate

# Export 

INPUT_TYPE = ['image_tensor', 'encoded_image_string_tensor', 'tf_example','float_image_tensor']



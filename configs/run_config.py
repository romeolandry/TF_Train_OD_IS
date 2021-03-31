"""TensorFlow provide an list of pre-trained Model available on 
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
for this project we choose "SSD RESNET V1 FPN 640x640" for Object detection and "Mask R-CNN Inception ResNet V2 1024x1024"
"""

## Tensorflow API 

Path_to_objection_dir = "/home/kamgo/Dokumente/Projects/Train_tensorflow_OD_API/models/research/object_detection"

## DataSet
COCO_YEARS= 2017
PATH_IMAGES = "images"
PATH_ANNOTATIONS="annotations"
PATH_TRAINED_MODELS = "models"
PREFIX_MODEL_NAME = "my_"
PATH_TO_LABELS_MAP = "Label_map/mscoco_complete_label_map.pbtxt"
PATH_TO_LABELS_TEXT = "Label_map/Label.txt"
PATH_TO_EXPORT_DIR = "exported_models"
SUFFIX_EXPORT = "_saved"
LogDir = "logdir"

TRACKED_OBJECT= ["person","car","motorcycle","tv","laptop","mouse","remote","keyboard","cell phone","bicycle"]

# metric
IOU_THRESHOLD = 0.5
SCORE_THRESHOLD = .2
DATA_SIZE_VALIDATION = 1.0 # float [0,1.0]
 
LIST_MODEL_TO_DOWNLOAD = {
    "ssd_resnet50_v1":"http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz",
    "maskrcnn":"http://download.tensorflow.org/models/object_detection/tf2/20200711/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8.tar.gz"
}

PATH_PRE_TRAINED_MODELS = "pre_trained_models"

## Train Configurations 

NUM_TRAIN_STEP = 200000
EVAL_ON_TRAIN_DATA = False # only supported in distributed training
SAMPLE_OF_N_EVAL = None
SAMPLE_OF_N_EVAL_ON_TRAIN = None # used when Eval on train data is true
CHECKPOINT_EVERY_N_STEP = 1000
RECORD_SUMMARY = True

# Evaluation 
CHECKPOINT_DIR = None # path to checkpoint of model to evaluate

# Export
INPUT_TYPE = ['image_tensor', 'encoded_image_string_tensor', 'tf_example','float_image_tensor']

# Convertor
PATH_TO_CONVERTED_MODELS = "converted_models"
PATH_KERAS_TO_TF = "tf_from_keras"

COLOR_PANEL = ["Salmon","Lime","LightCyan","LemonChiffon","PapayaWhip"]

# TF-TRT
MAX_WORKSPACE_SIZE_BITES = 1<<20
PRECISION_MODE = "FP32" # FP32 FP16 INT8 ['FP32', 'FP16', 'INT8', 'fp32', 'fp16', 'int8']
ACCEPTED_MODE = ['FP32', 'FP16', 'INT8', 'fp32', 'fp16', 'int8']
MIN_SEGMENTATION_SIZE = 3
GPU_MEM_CAP = None #  None to allow memory growth.
INPUT_TYPE_MODEL='int'

USE_GPU = True

PATH_PERFORMANCE_CONVERT = "performances/convertort.json"
PATH_PERFORMANCE_INFER = "performances"
PATH_DIR_IMAGE_INF = "images_inferences"

camere_input = 1
camera_width = 720
camera_height = 720


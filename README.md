# TF_Train_OD_IS

## Insatll Tensorflow Object detection API and COCO API for coco metric
## Download and Preprocess data
    - create TF-record file
    - provide an label-Map file `.pbtxt`
## Download the model and config pipeline
    - set the model checkpoint path  `fine_tune_checkpoint`
    - In to  train_input_reader set the path for `tf_record_input_reader` withe tf-record
    - in to eval_input_reader set the path for `tf_record_input_reader` for eval_input
## Train/Inference
## Convert to TensorRT
## infereance TRT-model
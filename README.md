# TF_Train_OD_IS

## Insatll Tensorflow Object detection API and COCO API for coco metric
## Download and Preprocess data
- create TF-record file:
    
    If you train on COCO and you have alredy download the dataset, make you create the directory `annotations` and `images` with is subdirectory

        Project_dir/
        ├─ ...
        ├─ annotations/
        │  └─ .json
        │  └─ record 
        ├─ images/
        |   └─train[years]
        |   └─test[years]
        |   └─val[years]
        ├─..
        └─ ...
    
    Run the following command to generate record file. If project doesn't content images and annotation, the coco dataset will be automaticly downloaded.

        python run.py --data_preprocessing
    

- provide an label-Map file `.pbtxt`: 
the directory Label_map content label-map for coco dataset

## Download the model and config pipeline
- set the model checkpoint path  `fine_tune_checkpoint`
- In to  train_input_reader set the path for `tf_record_input_reader` with the generate train-record tf-record
- in to eval_input_reader set the path for `tf_record_input_reader` for eval_input: give the path to test record. the model will evaluate durign training.

## Train/Inference
### Train

    python run.py -m maskrcnn


## Convert to TensorRT
## infereance TRT-model
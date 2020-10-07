# TF_Train_OD_IS

## Insatll Tensorflow Object detection API and COCO API for coco metric
### [Install Tensorflow Api](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html)

    ## Install tensorflow-gpu 2.x
    pip install tensorflow-gpu 

Clone [tensorflow api](https://github.com/tensorflow/models)

    cd Tensorflow/models/rearch/
    ## Protobuf insatllation
    protoc object_detection/protos/*.proto --python_out=.
    ## cocoApi
    pip install pycocotools
    ## 
    cp object_detection/packages/tf2/setup.py .
    python -m pip install .

Test installation

    # From within TensorFlow/models/research/
    python object_detection/builders/model_builder_tf2_test.py

you should see

        ...
    [       OK ] ModelBuilderTF2Test.test_create_ssd_models_from_config
    [ RUN      ] ModelBuilderTF2Test.test_invalid_faster_rcnn_batchnorm_update
    [       OK ] ModelBuilderTF2Test.test_invalid_faster_rcnn_batchnorm_update
    [ RUN      ] ModelBuilderTF2Test.test_invalid_first_stage_nms_iou_threshold
    [       OK ] ModelBuilderTF2Test.test_invalid_first_stage_nms_iou_threshold
    [ RUN      ] ModelBuilderTF2Test.test_invalid_model_config_proto
    [       OK ] ModelBuilderTF2Test.test_invalid_model_config_proto
    [ RUN      ] ModelBuilderTF2Test.test_invalid_second_stage_batch_size
    [       OK ] ModelBuilderTF2Test.test_invalid_second_stage_batch_size
    [ RUN      ] ModelBuilderTF2Test.test_session
    [  SKIPPED ] ModelBuilderTF2Test.test_session
    [ RUN      ] ModelBuilderTF2Test.test_unknown_faster_rcnn_feature_extractor
    [       OK ] ModelBuilderTF2Test.test_unknown_faster_rcnn_feature_extractor
    [ RUN      ] ModelBuilderTF2Test.test_unknown_meta_architecture
    [       OK ] ModelBuilderTF2Test.test_unknown_meta_architecture
    [ RUN      ] ModelBuilderTF2Test.test_unknown_ssd_feature_extractor
    [       OK ] ModelBuilderTF2Test.test_unknown_ssd_feature_extractor
    ----------------------------------------------------------------------
    Ran 20 tests in 68.510s

    OK (skipped=1)

## install requirement from this repository

    pip install -r requirement.txt

## Download and Preprocess data
- create TF-record file:
    
    If you train on COCO and you have alredy download the dataset, make you create the directory `annotations` and `images` as subdirectory

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
    
    Run the following command to generate record file. If project doesn't content images and annotation, the coco dataset will be automaticly downloaded and unpack.

        python run.py --data_preprocessing
    

- provide an label-Map file `.pbtxt`: 
the directory Label_map content label-map for coco dataset

## Train

### Download the model and config pipeline
- set the model checkpoint path  `fine_tune_checkpoint`
- In to  train_input_reader set the path for `tf_record_input_reader` with the generate train-record tf-record
- in to eval_input_reader set the path for `tf_record_input_reader` for eval_input: give the path to test record. the model will evaluate durign training.

Or run the following command to download and unpack the [available pre-trained]() model

    python run.py -m [model_name]

if the given model have already be downloaded, the training and evaluation will start, else it will be downlaoded.
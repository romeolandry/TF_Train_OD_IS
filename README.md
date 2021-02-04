# TF_Train_OD_IS

## Insatll Tensorflow Object detection API and COCO API for coco metric
### [Install Tensorflow Api](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html)

    ## Install tensorflow-gpu 2.x
    pip install tensorflow-gpu 

Clone [tensorflow api](https://github.com/tensorflow/models)

```shell
$ cd Tensorflow/models/research/
## Protobuf installation
$ protoc object_detection/protos/*.proto --python_out=.
## cocoApi
$ pip install pycocotools
## 
$ cp object_detection/packages/tf2/setup.py .
$ python -m pip install .
```

Test installation

```shell
# From within TensorFlow/models/research/
$ python object_detection/builders/model_builder_tf2_test.py
```

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

```shell
$ pip install -r requirement.txt
```

**Note:** Change the path to object_detection directory located inton `model/research/` 

```
# open run_configs 
# change Path_to_objection_dir
Path_to_objection_dir = absolut path to object_detection
```



## Download prepare data and Model

### Coco Dataset 

- create TF-record file:
  
    If you won to train on COCO and you have already download the dataset, make sure you create the directory `annotations` and `images` as subdirectory and create and symbolic link to your coco directory.

    ```shell
    # move into Project dir
    $ cd Project_dir
    # annotations
    $ ln -s path to annotation directory ./annotations
    # create images dir
    $ mkdir images
    # train test val
    $ ln -s path to test ./images/test2017
    $ ln -s path to val ./images/val2017
    $ ln -s path to train ./images/train2017
    ```
    

Run the following command to generate record file. If project doesn't content images and annotations directories, its will ask  if you wont to download  the coco dataset and unpack it.  say yes to continue with the download(it could take time) or no to abort and create the data directory manually.

- add --mask to create record for mask. else only record for box will be generate.

```shell
  $ python run.py --data_preprocessing
```

- provide an label-Map file `.pbtxt`: 
the directory Label_map content label-map for coco dataset

### Model

The `run_config.py` content a dictionary of models and her URL to automatically downloaded the model and unpack it.  run the following command to download a model and unpack it.

```shell
$ python run.py -m [model_name]
```

model_name could be ,, ***ssd_resnet50_v1***. in

- **`ssd_resnet50_v1`**: for ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz
- **`maskrcnn`**and: for mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8.tar.gz

**Note:** Training and test now only support **`ssd_resnet50_v1`**.

After the model have been downloaded, open the newly created directory `pre_trained_models`.

- move into the `model_dir/saved_mdel` and open `pipeline.config` file.

- set the model checkpoint path  `fine_tune_checkpoint` into the section `train_config`.

  `pre_trained_models/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0` . `fine_tune_checkpoint_type` give *detection* instead of *classification*

- In to  `train_input_reader` set the path for `tf_record_input_reader` with the generate train-record tf-record `annotations/coco_train.record-?????-of-00100` and the path of `label_map_path` e.g `Label_map/mscoco_complete_label_map.pbtxt`

- in to `eval_input_reader` set the path for `tf_record_input_reader`  e.g `annotations/coco_val.record-?????-of-00100` and the`label_map_path` .

After that the model are ready for the training or the inference. The Project directory structure look like.

```
Project_dir/
├─ ...
├─ annotations/
│  └─ .json
│  └─ .record 
├─ images/
|   └─train[years]
|   └─test[years]
|   └─val[years]
├─ ...
|
├─pre_trained_models
|	└─model_name
|		└checkpoint
|		└─ saved_model
|		└─pipeline.config
├─..
└─ ...
```



## Train

if the data configuration and model configuration was well done, the following command will launch the train phase

```shell
$ python run.py -m ssd_resnet50_v1
```

At the end of the train, the train model will be saved into a new directory `model` and  will also be exported as a definition graph into `exported_model` in case you won to continue train the same model.

## Evaluation

## Inference

The file `run_inference` will be use to apply the model(pre-trained and exported) on saved images or from web-cam.

- `--webcam ` if you won to use web-cam module 
- `-p or path_to_images` to set the directory contenting images the default directory  is the `test2017`.
- `-t or --type` to choose between an SSD-Resnet50-v1(ssd) and MASK-RCNN(ssd)
- `-s or --size` Input size of the model.
-  `-i or --nb_img` to set the number of image you won to inference into th given directory.
- `-m or --model` to set the path to the model directory. in case of inference on image path to `saved_model`directory else path to model.
- `-l or --label` set the path to label.

To run inference on images.

```shell
$ python run_inference.py -i 3 -s 640 -t mask -m path_to_saved_model_dir
```

It will create a directory named `images_inferences` to save inference.

To inference using web-cam run :

```shell
$ python run_inference.py --webcam  -t mask -m path_to_model_name
```
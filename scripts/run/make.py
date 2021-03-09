import sys
import os
import subprocess
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = "2,4"

sys.path.append(os.path.abspath(os.curdir))

from configs.run_config import *
from object_detection import *
#from scripts.api_scrpit import *

'''    
    Return: True if proceed without error
            False else.
    The path to cloned model/research(object_detection is set into run_config
'''

def make_preprocessing (type):

    # get create_coco_tf_record.py path
    create_coco_tf_record = os.path.join(Path_to_objection_dir,'dataset_tools/create_coco_tf_record.py')
    if not os.path.exists(create_coco_tf_record):
        sys.stderr.write("Please set the correct path to object_detection dir in the file run_config")
    command = 'python ' + create_coco_tf_record

    Train_images = os.path.join(PATH_IMAGES,'train'+str(COCO_YEARS))
    Test_Images = os.path.join(PATH_IMAGES,'test'+str(COCO_YEARS))
    Val_img = os.path.join(PATH_IMAGES,'val'+str(COCO_YEARS))

    Train_ann = os.path.join(PATH_ANNOTATIONS,'instances_train'+str(COCO_YEARS)+ '.json')
    Test_ann = os.path.join(PATH_ANNOTATIONS,'image_info_test-dev'+str(COCO_YEARS)+ '.json')
    Val_ann = os.path.join(PATH_ANNOTATIONS,'instances_val'+str(COCO_YEARS)+ '.json')

    # set output directory
    output = os.path.join(PATH_ANNOTATIONS,type)
    if not os.path.exists(output):
        os.mkdir(output)
    if type == "mask":
        arguments = ' --logtostder --include_masks --train_image_dir '+ Train_images +' --val_image_dir '\
            + Val_img +' --test_image_dir '+ Test_Images + ' --train_annotations_file '+ Train_ann + \
                ' --val_annotations_file ' + Val_ann + ' --testdev_annotations_file '+ Test_ann + \
                    ' --output_dir '+ output
    else:
        arguments = ' --logtostder --train_image_dir '+ Train_images +' --val_image_dir '\
            + Val_img +' --test_image_dir '+ Test_Images + ' --train_annotations_file '+ Train_ann + \
                ' --val_annotations_file ' + Val_ann + ' --testdev_annotations_file '+ Test_ann + \
                    ' --output_dir '+ output


    run_create_coco_record_script_file = command +  arguments
    try:
        subprocess.call(run_create_coco_record_script_file, shell= True)
        return True
    except expression as identifier:
        pass
    return False

def make_eval(model_dir, pipeline, checkpoint,timeout):
        
    # get create_coco_tf_record.py path
    model_main_tf2 = os.path.join(Path_to_objection_dir,'model_main_tf2.py')
    if not os.path.exists(model_main_tf2):
        sys.stderr.write("Please set the correct path to object_detection dir in the file run_config")
    command = 'python ' + model_main_tf2

    arguments = ' --model_dir='+ model_dir +' --pipeline_config_path='+ pipeline + \
        ' --num_train_steps='+ str(NUM_TRAIN_STEP) +' --checkpoint_dir=' + checkpoint + \
        ' --eval_timeout' +  str(timeout)+ ' --alsologtostderr'

    try:
        subprocess.call(command + arguments, shell= True)
        return True
    except subprocess.CalledProcessError as exc:
        print("Status : FAIL", exc.returncode, exc.output)
        return False

def make_train(model_name):
    """
    Execute the model_main_tf2.py provided by the Api to train an selected model.
    the trained model will be save into model/ directory. this will be created if not exist.
    """
    # check if directory exist
    model_url = LIST_MODEL_TO_DOWNLOAD[model_name] # get model url from
    file_name = (model_url.split("/")[-1]).split(".")[0] # file name from url
    file_name = file_name.split('.')[0] # get file name without extension

    new_file_name = PREFIX_MODEL_NAME + file_name # add prefixe
    
    if not os.path.exists(os.path.join(PATH_TRAINED_MODELS,new_file_name)):
        if not os.path.exists(PATH_TRAINED_MODELS):
            os.mkdir(PATH_TRAINED_MODELS)
        os.mkdir(os.path.join(os.path.join(PATH_TRAINED_MODELS,new_file_name)))
    
    ##  get pipeline path
    if not os.path.isfile(os.path.join(PATH_PRE_TRAINED_MODELS,file_name,'pipeline.config')):
        assert "Pre-trained model don't have config"
    
    path_to_pipeline = os.path.join(PATH_PRE_TRAINED_MODELS,file_name,'pipeline.config')
    path_to_save_trained_model = os.path.join(PATH_TRAINED_MODELS,new_file_name)

    # get create_coco_tf_record.py path
    model_main_tf2 = os.path.join(Path_to_objection_dir,'model_main_tf2.py')
    if not os.path.exists(model_main_tf2):
        sys.stderr.write("Please set the correct path to object_detection dir in the file run_config")
    command = 'python ' + model_main_tf2

    arguments = ' --model_dir='+ path_to_save_trained_model +' --pipeline_config_path='+ path_to_pipeline + \
        ' --num_train_steps='+ str(NUM_TRAIN_STEP) + ' --checkpoint_every_n='+ str(CHECKPOINT_EVERY_N_STEP) + \
        ' --alsologtostderr'
    try:
        subprocess.call(command + arguments, shell= True)
        return True
    except subprocess.CalledProcessError as exc:
        print("Status : FAIL", exc.returncode, exc.output)
        return False

def make_export(model_dir, pipeline, checkpoint):
    """
    Execute the exporter_main_v2.py provided by the Api to train an selected model.
    the trained model will be save into model/ directory. this will create if not exist.
    """
    # check if directory exist

    # get create_coco_tf_record.py path
    exporter_main_v2 = os.path.join(Path_to_objection_dir,'exporter_main_v2.py')
    if not os.path.exists(exporter_main_v2):
        sys.stderr.write("Please set the correct path to object_detection dir in the file run_config")
    command = 'python ' + exporter_main_v2

    arguments =' --input_type '+ INPUT_TYPE[0] + ' --pipeline_config_path=' + pipeline + ' --trained_checkpoint_dir=' \
        + checkpoint + ' --output_directory=' +  model_dir

    try:
        subprocess.call(command + arguments, shell= True)
        return True
    except subprocess.CalledProcessError as exc:
        print("Status : FAIL", exc.returncode, exc.output)
        return False

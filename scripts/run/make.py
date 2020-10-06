import sys
import os
import subprocess

sys.path.append(os.path.abspath(os.curdir))

from configs.run_config import *
from scripts.api_scrpit import *

def make_preprocessing ():
    
    """
        This function run create_coco_tf_record.py as subprocess
        Return: True if proceed without error
            False else.
    """
    command = 'python scripts/api_scrpit/create_coco_tf_record.py '

    Train_imgs = os.path.join(PATH_IMAGES,'train'+str(COCO_YEARS))
    Test_Imgs = os.path.join(PATH_IMAGES,'test'+str(COCO_YEARS))
    Val_img = os.path.join(PATH_IMAGES,'val'+str(COCO_YEARS))

    Train_ann = os.path.join(PATH_ANNOTATIONS,'instances_train'+str(COCO_YEARS)+ '.json')
    Test_ann = os.path.join(PATH_ANNOTATIONS,'image_info_test-dev'+str(COCO_YEARS)+ '.json')
    Val_ann = os.path.join(PATH_ANNOTATIONS,'instances_val'+str(COCO_YEARS)+ '.json')

    arguments = '--logtostder --train_image_dir '+ Train_imgs +' --val_image_dir '\
        + Val_img +' --test_image_dir '+ Test_Imgs + ' --train_annotations_file '+ Train_ann + \
            ' --val_annotations_file ' + Val_ann + ' --testdev_annotations_file '+ Test_ann + \
                ' --output_dir '+ os.path.join(PATH_ANNOTATIONS) 

    run_create_coco_record_script_file = command +  arguments
    try:
        subprocess.call(run_create_coco_record_script_file, shell= True)
        return True
    except expression as identifier:
        pass
    return False

def make_train(model_name):
    """
    Execute the model_main_tf2.py provided by the Api to train an selected model.
    the trained model will be save into model/ directory. this will create if not exist.
    """

    # check if dirctory exist
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
    
    command = 'python scripts/api_scrpit/model_main_tf2.py '
    arguments = '--model_dir='+ path_to_save_trained_model +' --pipeline_config_path='+ path_to_pipeline + \
        ' --num_train_steps='+ str(NUM_TRAIN_STEP) + ' --checkpoint_every_n=' + str(CHECKPOINT_EVERY_N_STEP)
    try:
        
        output = subprocess.check_output(
            command + arguments,
            stderr=subprocess.STDOUT,
            shell= True,
            universal_newlines= True
        )
    except subprocess.CalledProcessError as exc:
        print("Status : FAIL", exc.returncode, exc.output)
        return False
    else:
        print("Output: \n{}\n".format(output))
    return True

def make_export(model_name):
    """
    Execute the model_main_tf2.py provided by the Api to train an selected model.
    the trained model will be save into model/ directory. this will create if not exist.
    """
    # check if dirctory exist
    model_url = LIST_MODEL_TO_DOWNLOAD[model_name] # get model url from
    file_name = (model_url.split("/")[-1]).split(".")[0] # file name from url
    file_name = file_name.split('.')[0] # get file name without extension

    file_name_trained = PREFIX_MODEL_NAME + file_name

    new_file_name = PREFIX_MODEL_NAME + file_name + SUFIX_EXPORT # add prefixe
    
    if not os.path.exists(os.path.join(PATH_TRAINED_MODELS,file_name_trained)):
        return False
    trained_checkpoint_dir = os.path.join(PATH_TRAINED_MODELS,file_name_trained)
    # get pipeline 
    
    path_to_pipeline = os.path.join(PATH_PRE_TRAINED_MODELS,file_name,'pipeline.config')

    # check output dir create if not exist
    if not os.path.exists(os.path.join(PATH_TO_EXPORT_DIR,new_file_name)):
        if not os.path.exists(PATH_TO_EXPORT_DIR):
            os.mkdir(PATH_TO_EXPORT_DIR)
        os.mkdir(os.path.join(PATH_TO_EXPORT_DIR,new_file_name))
    
    output_dir = os.path.join(PATH_TRAINED_MODELS,new_file_name)
    command = 'python scripts/api_scrpit/exporter_main_v2.py '
    arguments =' --pipeline_config_path=' + path_to_pipeline + ' --trained_checkpoint_dir=' \
        + trained_checkpoint_dir + ' --output_directory=' +  output_dir
    
    try:
        
        output = subprocess.check_output(
            command + arguments,
            stderr=subprocess.STDOUT,
            shell= True,
            universal_newlines= True
        )
    except subprocess.CalledProcessError as exc:
        print("Status : FAIL", exc.returncode, exc.output)
        return False
    else:
        print("Output: \n{}\n".format(output))
    return True

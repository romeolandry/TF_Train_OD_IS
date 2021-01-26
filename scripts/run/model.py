import os
import sys
import time
import click

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
sys.path.append(os.path.abspath(os.curdir))

import tensorflow as tf

from tensorflow import keras
from pathlib import Path
from object_detection.utils import config_util
from object_detection.builders import model_builder
from configs.run_config import *

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

class Model:
    def __init__(self,
                 path_to_model,
                 checkpoint = 'ckpt-0'):
        self.__path_to_model = path_to_model
        self.__checkpoint = checkpoint
        self.__detection_model = None

        ## get model name from path
        if os.path.basename(self.__path_to_model) != 'saved_model':
            self.__model_name = os.path.basename(self.__path_to_model)
        else:
            self.__model_name = self.__path_to_model.split('/')[-2]

    def Load_model(self):
        """
            This function load the from a given path.
            the given model can be an keras model or an tensorflow model.
            If a keras-model is given it will converted in to a tensorflow model saved and load as tensorflow model

            Arg: Path to model directory for Tensorflow or .h5 for keras

            Return: model
        """
        # check if is keras .h5 oder a tensorflow model
        elapstime = 0
        if Path(self.__path_to_model).suffix == '.h5':
            click.echo(click.style(f"\n keras model will be loaded \n", bold=True, fg='green'))
            keras_model = keras.models.load_model(self.__path_to_model)
            path_to_convert = os.path.join(PATH_KERAS_TO_TF,self.__model_name)
            tf.saved_model.save(keras_model,path_to_convert)
            self.__path_to_model = path_to_convert
        
        try:
            click.echo(click.style(f"\n tensorflow frozen graph will be loaded. \n", bold=True, fg='green'))
            start_time = time.time()
            self.__detection_model = tf.saved_model.load(self.__path_to_model)
            end_time = time.time()
            elapstime = end_time - start_time

        except FileExistsError:
            raise(f"The save model {self.__path_to_model} can't be loaded!")

        # return tensorflow frozen graph
        click.echo(click.style(f"\n the model was loaded in {elapstime} seconds. \n", bold=True, fg='green'))
        return self.__detection_model, self.__model_name

    
    '''
        Build model from a checkpoint 
    '''
    def build_detection_model(self):

        if not os.path.isfile (os.path.join(self.__path_to_model,'pipeline.config')):
            exit("the model don't content configuration")
        
        click.echo(click.style(f"\n Build model from checkpoint.... \n", bold=True, fg='green'))

        start_time = time.time()
        # Load pipeline config and build a detection model
        configs = config_util.get_configs_from_pipeline_file(os.path.join(self.__path_to_model,'pipeline.config'))
        model_config = configs['model']
        self.__detection_model = model_builder.build(model_config=model_config, is_training=False)

        end_time = time.time()
        # Restore checkpoint pre_trained_models/ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8/checkpoint
        if not os.path.isdir(os.path.join(self.__path_to_model,'checkpoint/')):
            exit("there isn't checkpoint in to given directory")
        
        ckpt = tf.compat.v2.train.Checkpoint(model=self.__detection_model)
        ckpt.restore(os.path.join(self.__path_to_model,'checkpoint/' + self.__checkpoint)).expect_partial()

        elapstime = end_time - start_time
        click.echo(click.style(f"\n The model was builded in {end_time - start_time} \n", bold=True, fg='green'))
        return self.__detection_model, self.__model_name
    

import os 
import sys
import json

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from configs.run_config import *
from tensorflow import keras
from tensorflow.python.compiler.tensorrt import trt_convert as trt

from pathlib import Path
import click


sys.path.append(os.path.abspath(os.curdir))


class Convertor:

    def __init__(self,
                 path_to_model,
                 precision_mode='PRECISION_MODE',
                 max_workspace_size_bytes = 'MAX_WORKSPACE_SIZE_BITES'):

        self.__path_to_model = path_to_model
        self.__precision_mode = precision_mode
        self.__max_workspacesize_byte = max_workspace_size_bytes

        if os.path.exists(self.__path_to_model):
            self.__model_name = os.path.abspath(self.__path_to_model).split('/')[-2]
        else:
            raise(f"file or diretory {self.__path_to_model} doesn'\n t exist")
    
    ''' This function help to load model:
        if is an keras model it will converted to tf frozen
        else it the tf frozen model will be loaded
        Return: model
    '''
    def Load_model(self):
    
        # check if is keras .h5 oder a tensorflow model
        if Path(self.__path_to_model).suffix == '.h5':
            click.echo(click.style(f"\n keras model will be loaded \n", bold=True, fg='green'))
            keras_model = self.keras_to_frozen_graph()
            path_to_convert = os.path.join(PATH_KERAS_TO_TF,self.__model_name)
            tf.saved_model.save(keras_model,path_to_convert)
            self.__path_to_model = path_to_convert
        
        try:
            click.echo(click.style(f"\n tensorflow frozen graph will be loaded. \n", bold=True, fg='green'))
            model = tf.saved_model.load(self.__path_to_model)
        except:
            return False

        # return tensorflow frozen graph
        return model
    
    """
        This funcotion will load a Keras Model and convert it to 
        an Tensorflow frozen graph .pb model
    """
    def keras_to_frozen_graph(self):
        try:
            model = tf.keras.models.load_model(self.__path_to_model)
        except:
            return False
        return model

    ''' 
        Check if keras model is really a keras model
        and if tf model directory conten .pb file
    '''
    def check_model_path(self):

        # check if is keras .h5 oder a tensorflow model
        if Path(self.__path_to_model).suffix == '.h5':
            keras_model = self.keras_to_frozen_graph()
            path_to_convert = os.path.join(PATH_KERAS_TO_TF,self.__model_name)
            tf.saved_model.save(keras_model,path_to_convert)
            # change model model name
            self.__model_name = os.path.splitext(os.path.basename(self.__path_to_model))[0]
            self.__path_to_model = path_to_convert

        # check integrity of model
        if not (fname.endswith('.pb') for fname in os.listdir(self.__path_to_model)):
            sys.stderr.write(f"the model don't content .pb file. Please make sure \n the directory {self.__path_to_model} is an tF model")

        # return tensorflow frozen graph
        return self.__path_to_model

    ''' 
        Convert all tensorflow model to tensorflow-tensorRT model
        Tf-Model could be:
            - saved_model_dir (vairables dir,assets,.pd-File)
            - Freeze model (frozen_model.pb)
        the converted model will be save into 'converted_models' directory
    '''
    def convert_to_TF_TRT_graph_and_save(self,calibration_data = None):
        assert self.__precision_mode in ['FP32', 'FP16', 'INT8', 'fp32', 'fp16', 'int8'], f" the given precision mode {self.__precision_mode} not supported.\n It should be one of {['FP32', 'FP16', 'INT8', 'fp32', 'fp16', 'int8']}"
        
        original_name = self.__path_to_model
        
        # Check which model have be given
        self.__path_to_model = self.check_model_path()

        if self.__precision_mode in ['FP32','fp32']:
            self.__model_name = self.__model_name + '_TFTRT_F32'

        if self.__precision_mode in ['FP16','fp16']:
            self.__model_name = self.__model_name + '_TFTRT_F16'

        if self.__precision_mode in ['INT8','int8']:
            if calibration_data == None:
                raise('calibraion is required to process this convertion')
            self.__model_name = self.__model_name + '_TFTRT_INT8'
        
        output_saved_model_dir = os.path.join(PATH_TO_CONVERTED_MODELS,self.__model_name)

        conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
            precision_mode=self.__precision_mode,
            max_workspace_size_bytes=self.__max_workspacesize_byte
        )

        converter = trt.TrtGraphConverterV2(
            input_saved_model_dir=self.__path_to_model,
            conversion_params=conversion_params
        )
        click.echo(click.style(f"\n Using precision mode: {self.__precision_mode}\n", bold=True, fg='green'))

        if self.__precision_mode == trt.TrtPrecisionMode.INT8:
            def calibraion_input_fn():
                yield(calibration_data,)
            converter.convert(calibraion_input_fn)
        else:
            converter.convert()
        
        click.echo(click.style(f"\n Saving {self.__model_name} \n", bold=True, fg='green'))
        converter.save(output_saved_model_dir = output_saved_model_dir)
        click.echo(click.style(f"\n Complet \n", bold=True, fg='green'))
        
        return original_name,output_saved_model_dir
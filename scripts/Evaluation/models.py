import os
import sys
import time
import click

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
sys.path.append(os.path.abspath(os.curdir))

import tensorflow as tf
import numpy as np
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow import keras
from pathlib import Path
from tensorflow.python.compiler.tensorrt import trt_convert as trt
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
        self.__model_name = self.__path_to_model.split('/')[-2]
        self.__logdir = os.path.join(LogDir,self.__model_name)

    def tf_pb_viewer(self):
        if not (self.check_model_path()):
            sys.stderr.write("Load an correct model file")
        if not os.path.exists(self.__logdir):
            os.makedirs(self.__logdir)
        
        graph_def, model = self.load_freezed_model()
        
        with tf.compat.v1.Session() as sess:
            sess.graph.as_default()
            tf.import_graph_def(graph_def)
            train_writer = writer = tf.compat.v1.summary.FileWriter(self.__logdir,sess.graph)
            train_writer.flush()
            train_writer.close()

        
    ''' 
        Check if keras model is really a keras model
        and if tf model directory conten .pb file
    '''
    def check_model_path(self):
        if os.path.isdir(self.__path_to_model):
            return (fname.endswith('.pb') for fname in os.listdir(self.__path_to_model))
        else:
            return os.path.exists(self.__path_to_model)
        
    
    ''' 
        This function load the from a given path.
        Arg: Path to savedModel directory
        Return: model
    '''
    def Load_savedModel_model(self):
        # check if is keras .h5 oder a tensorflow model
        elapstime = 0
        if self.check_model_path():        
            try:
                click.echo(click.style(f"\n tensorflow SavedModel will be loaded. \n", bold=True, fg='green'))
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
        Load freezed model using tensorflow 1.x read function
        tf 2.X don't use Graph and session anymore.
    '''
    def load_freezed_model(self):
        elapstime = 0
        if not (self.check_model_path()):
            sys.stderr.write("Load an correct model file")
        
        with tf.io.gfile.GFile(self.__path_to_model,'rb') as f:
            start =time.time()
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            end = time.time()
            elapstime = end - start
        
        click.echo(click.style(f"\n the was parsed in  {elapstime} seconds. \n", bold=True, fg='green'))
        return graph_def, self.__model_name
                

class Convertor:
    
    def __init__(self,
                 path_to_model,
                 precision_mode='PRECISION_MODE',
                 max_workspace_size_bytes = 'MAX_WORKSPACE_SIZE_BITES'):

        self.__path_to_model = path_to_model
        self.__precision_mode = precision_mode
        self.__max_workspacesize_byte = max_workspace_size_bytes


        model = Model(self.__path_to_model)
        model_for_detection, model_name = model.Load_savedModel_model()

        self.__model_name = model_name
        self.__model = model_for_detection
    
    ''' 
        Convert all tensorflow model to tensorflow-tensorRT model
        Tf-Model could be:
            - saved_model_dir (vairables dir,assets,.pb-File)
        the converted model will be save into 'converted_models' directory
    '''
    def convert_to_TF_TRT_graph_and_save(self,calibration_data = None):
        assert self.__precision_mode in ['FP32', 'FP16', 'INT8', 'fp32', 'fp16', 'int8'], f" the given precision mode {self.__precision_mode} not supported.\n It should be one of {['FP32', 'FP16', 'INT8', 'fp32', 'fp16', 'int8']}"
        
        original_name = self.__path_to_model

        if self.__precision_mode in ['FP32','fp32']:
            self.__model_name = self.__model_name + '_TFTRT_F32'

        if self.__precision_mode in ['FP16','fp16']:
            self.__model_name = self.__model_name + '_TFTRT_F16'

        if self.__precision_mode in ['INT8','int8']:
            if calibration_data == None:
                raise('calibraion is required to process this convertion')
            self.__model_name = self.__model_name + '_TFTRT_INT8'
        
        output_saved_model_dir = os.path.join(PATH_TO_CONVERTED_MODELS,self.__model_name + '/saved_model')

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

    ''' 
        Freeze Tensorflow savedModel for Inference 
    '''
    
    def freeze_savedModel(self, image_size=640):

        infer = self.__model.signatures["serving_default"]
      
        # convert the model to ConcreteFunction.
        # since tf.2.x doesn't session any more.
        full_model = tf.function(lambda input_tensor: infer(input_tensor))
        
        full_model = full_model.get_concrete_function(
                            input_tensor=tf.TensorSpec(shape=[1, image_size, image_size, 3], 
                                                        dtype=np.uint8)
                                                        ) 
        frozen_func = convert_variables_to_constants_v2(full_model)
        frozen_func.graph.as_graph_def()

        
        input_layers = [x.name for x in frozen_func.inputs]
        outputs = [x.name for x in frozen_func.outputs]
        
        # Save specification
        model_dir = os.path.join(self.__path_to_model,'../forzen_model')
        
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        with open(os.path.join(model_dir,'about.txt'), "w+") as f:
            f.write("*****************************Original input signature *************************** \n")
            f.write("\n input \n")
            for value in input_layers:
                f.write("\n Original output signature \n")
            for key,value in infer.structured_outputs.items():
                f.write(f"\n {key} ={value}\n")
            
            f.write("\n *************************** freezed signature ********************* \n")
            f.write("\n input \n")
            for value in input_layers:
                f.write(f"\n {value}\n")
            f.write("\n output \n")
            for value in outputs:
                if  value == "Identity:0":
                    f.write(f"\n detection_anchor_indices ==> {value}  \n")
                if  value == "Identity_1:0":
                    f.write(f"\n detection_boxes ==> {value}  \n")

                if  value == "Identity_2:0":
                    f.write(f"\n detection_classes ==> {value}  \n")

                if  value == "Identity_3:0":
                    f.write(f"\n detection_multiclass_scores ==> {value}  \n")

                if  value == "Identity_4:0":
                    f.write(f"\n detection_scores ==> {value}  \n")

                if  value == "Identity_5:0":
                    f.write(f"\n num_detections ==> {value}  \n") 

                if  value == "Identity_6:0":
                    f.write(f"\n raw_detection_boxes ==> {value}  \n")

                if  value == "Identity_7:0":
                    f.write(f"\n raw_detection_scores ==> {value}  \n")

        # Save the freezed Graph
        tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                          logdir=model_dir,
                          name="frozen_graph.pb",
                          as_text=False)
        click.echo(click.style(f"\n model was freezed and saved to {model_dir}\n", bold=True, fg='green'))

        
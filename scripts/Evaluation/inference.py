import os
import sys
import tensorflow as tf
import time
import click
import matplotlib.pyplot as plt
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

sys.path.append(os.path.abspath(os.curdir))

from configs.run_config import *
from scripts.api_scrpit import *
from scripts.Evaluation.utils import *
from pathlib import Path

from scripts.api_scrpit import label_map_util
from scripts.api_scrpit import visualization_utils as viz_utils

from object_detection.utils import config_util
from object_detection.builders import model_builder

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.get_logger().setLevel('ERROR')

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)



class Inefrence :
    def __init__(self,
                 path_to_images,
                 path_to_model,
                 path_to_labels,
                 checkpoint = 'ckpt-0'):
        self.__path_to_images = path_to_images
        self.__path_to_model = path_to_model
        self.__path_to_labels = path_to_labels
        self.__checkpoint = checkpoint
        self.__detection_model = None

        ## create category index for coco 
        self.__category_index = label_map_util.create_category_index_from_labelmap(self.__path_to_labels,use_display_name=True)

        if os.path.basename(self.__path_to_model) != 'saved_model':
            self.__images_name_prefix = os.path.basename(self.__path_to_model)
        else:
            self.__images_name_prefix = self.__path_to_model.split('/')[-2]

    def Load_model(self):
        """
            This function load the from a given path.
            the given model can be an keras model or an tensorflow model.
            If a keras-model is given it will converted in to a tensorflow model saved and load as tensorflow model

            Arg: Path to modelö directory for Tensorflow or .h5 for keras

            Return: model
        """
        # check if is keras .h5 oder a tensorflow model
        elapstime = 0
        if Path(self.__path_to_model).suffix == '.h5':
            click.echo(click.style(f"\n keras model will be loaded \n", bg='green', bold=True, fg='white'))
            keras_model = self.keras_to_frozen_graph()
            path_to_convert = os.path.join(PATH_KERAS_TO_TF,self.__model_name)
            tf.saved_model.save(keras_model,path_to_convert)
            self.__path_to_model = path_to_convert
        
        try:
            click.echo(click.style(f"\n tensorflow frozen grahp will loaded. \n", bg='green', bold=True, fg='white'))
            start_time = time.time()
            model = tf.saved_model.load(self.__path_to_model)
            end_time = time.time()
            elapstime = end_time - start_time

        except FileExistsError:
            raise(f"The save model {self.__path_to_model} can't be loaded!")

        # return tensorflow frozen graph
        click.echo(click.style(f"\n the model was loaded in {elapstime} seconds. \n", bg='green', bold=True, fg='white'))
        return model

    def infernce_images_from_dir(self, number_of_images=None):

        images_list = []
        print(f"loading {number_of_images} images from {self.__path_to_images}")
        images_list = load_img_from_folder(self.__path_to_images,number_of_images)
        # get loaded model
        print("loading model")
        model = self.Load_model()
        print("terminate")

        i = 0
        for image_np in images_list:
            # convert images to be a tensor
            input_tensort = tf.convert_to_tensor(image_np)
            input_tensort = input_tensort[tf.newaxis,...]

            detections = model(input_tensort)
            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
                          for key, value in detections.items()}

            detections['num_detections'] = num_detections


            # detection_classes should be ints.
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
            image_np_with_detections = image_np.copy()

            viz_utils.visualize_boxes_and_labels_on_image_array(image_np_with_detections,
                                                                detections['detection_boxes'],
                                                                detections['detection_classes'],
                                                                detections['detection_scores'],
                                                                self.__category_index,
                                                                use_normalized_coordinates=True,
                                                                max_boxes_to_draw=200,
                                                                min_score_thresh=.30,
                                                                agnostic_mode=False)
                

            plt.figure()
            plt.imshow(image_np_with_detections)
            if not os.path.isdir(PATH_DIR_IMAGE_INF):
                os.mkdir(PATH_DIR_IMAGE_INF)
            plt.savefig(os.path.join(PATH_DIR_IMAGE_INF,self.__images_name_prefix + "_" + str(i)))
            i +=1
        print('Done')
            #plt.show()

    def build_detection_model(self):

        if not os.path.isfile (os.path.join(self.__path_to_model,'pipeline.config')):
            raise("the model dont content configuration")

        # Load pipeline config and build a detection model
        configs = config_util.get_configs_from_pipeline_file(os.path.join(self.__path_to_model,'pipeline.config'))
        model_config = configs['model']
        self.__detection_model = model_builder.build(model_config=model_config, is_training=False)

        # Restore checkpoint pre_trained_models/ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8/checkpoint
        if not os.path.isdir(os.path.join(self.__path_to_model,'checkpoint/')):
            raise("there isn't checkpoint in to given driectory")
        
        ckpt = tf.compat.v2.train.Checkpoint(model=self.__detection_model)
        ckpt.restore(os.path.join(self.__path_to_model,'checkpoint/' + self.__checkpoint)).expect_partial()

    @tf.function
    def detect_fn(self,image):

        """Detect objects in image."""
        image, shapes = self.__detection_model.preprocess(image)
        prediction_dict = self.__detection_model.predict(image, shapes)
        detections = self.__detection_model.postprocess(prediction_dict, shapes)

        return detections, prediction_dict, tf.reshape(shapes, [-1])


    def inference_from_wedcam_with_checkpoint(self):

        # build the model 
        self.build_detection_model()

        cap = cv2.VideoCapture(0)

        while True:
            # Read frame from camera
            ret, image_np = cap.read()

            # expand image to have shape :[1, None, None,3]
            image_np_expanded = np.expand_dims(image_np, axis=0)

            input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0),dtype=tf.float32)
            detections, predictions_dict, shapes = self.detect_fn(input_tensor)

            label_id_offset = 1
            image_np_with_detections = image_np.copy()

            viz_utils.visualize_boxes_and_labels_on_image_array(image_np_with_detections,
                                                                detections['detection_boxes'][0].numpy(),
                                                                (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
                                                                detections['detection_scores'][0].numpy(),
                                                                self.__category_index,
                                                                use_normalized_coordinates=True,
                                                                max_boxes_to_draw=200,
                                                                min_score_thresh=.30,
                                                                agnostic_mode=False)


            # Display output
            cv2.imshow('object detection', cv2.resize(image_np_with_detections, (800, 600)))

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
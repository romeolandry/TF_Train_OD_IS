import sys
import os
import subprocess

sys.path.append(os.path.abspath(os.curdir))

from configs.run_config import *
from scripts.prepreocessing import create_coco_tf_record

def do_create_coco_record ():
    command = 'python scripts/prepreocessing/create_coco_tf_record.py '

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
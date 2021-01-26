import os
import sys
import click

from urllib.parse import urlparse
import zipfile
from tqdm.auto import tqdm
import requests
import tarfile
import shutil


sys.path.append(os.path.abspath(os.curdir))

import configs.run_config as cfg


def save_zip_from_url(url, save_dir):

    if not os.path.exists(os.path.join(os.path.abspath(os.curdir),save_dir)):
        os.mkdir(os.path.join(os.path.abspath(os.curdir),save_dir))
    
    zip_filename_location = os.path.join(os.path.abspath(os.curdir),save_dir,os.path.basename(urlparse(url).path))
    
    # save zip
    try:
        resp=requests.get(url, stream=True)
        total = int(resp.headers.get('content-length', 0))

        with open (zip_filename_location, 'wb') as file, tqdm(
            desc=zip_filename_location,
            total= total,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in resp.iter_content(chunk_size=4096):
                size = file.write(data)
                bar.update(size)
        return zip_filename_location
    except expression as identifier:
        pass

def Extract_zip_file (path_to_zip,dir_to_save_into):
    """
    Extract zip conten and change the structure of directory to match our required structure.
    """
    with zipfile.ZipFile(path_to_zip) as zf:
        
        for member in tqdm(zf.namelist(), desc='Extracting'):
            try:
                if ('annotations' in member) and (member.endswith('.json')):                    
                    zf.extract(member, dir_to_save_into)
                    shutil.move(os.path.join(dir_to_save_into,member),dir_to_save_into)
                if ('train' in member):
                    zf.extract(member, dir_to_save_into)
                if ('test' in member):
                    zf.extract(member, dir_to_save_into)
                if ('val' in member):
                    zf.extract(member, dir_to_save_into)
            except zipfile.error as e:
                pass

    #delete zip
    os.remove(path_to_zip)
    if(os.path.isdir(os.path.join(dir_to_save_into,'annotations'))):
        # remove the tmp annotations directory
        shutil.rmtree(os.path.join(dir_to_save_into,'annotations'))

def download_pre_trained_model (model_name):
    """
        This function check if the choose model have already been downloaded.
        when not the model will been Downloaded and extracted: return 0
        when yes it return 1.
        when an error occur it return -1
    """
    model_url = cfg.LIST_MODEL_TO_DOWNLOAD[model_name]

    file_name = (model_url.split("/")[-1]).split(".")[0] # get file name from url

    # create directory if not exit
    if not os.path.exists(os.path.join(cfg.PATH_PRE_TRAINED_MODELS,file_name)):
        if not os.path.exists(cfg.PATH_PRE_TRAINED_MODELS):
            os.mkdir(cfg.PATH_PRE_TRAINED_MODELS)
        try:
            click.echo(click.style(f"\n Downloading the {file_name} model \n", bg='green', bold=True, fg='white'))
            file_location = save_zip_from_url(model_url, cfg.PATH_PRE_TRAINED_MODELS)
    
            click.echo(click.style(f'\n Extraction of {file_name} ...\n',  bg='blue', bold=True, fg='white'))

            # Extract tar-file to annotation directory
            with tarfile.open(name=file_location) as tar:
                for member in tqdm(iterable=tar.getmembers(), total=len(tar.getmembers())):
                    tar.extract(member=member, path=cfg.PATH_PRE_TRAINED_MODELS)
            # delete tar-file
            os.remove(file_location)
            return 0

        except expression as identifier:
            return -1
    else:
        click.echo(click.style(f"\n {file_name} have been downloaded \n", bg='blue', bold=True, fg='white'))
        return 1


def download_coco():
    """
        this code will download coco dataset required image Train test val and annotation from 2017.
        - http://images.cocodataset.org/zips/train2017.zip  
        - http://images.cocodataset.org/zips/val2017.zip
        - http://images.cocodataset.org/annotations/annotations_trainval2017.zip

        - http://images.cocodataset.org/zips/test2017.zip
        - http://images.cocodataset.org/annotations/image_info_test2017.zip
    """ 
    file_type = '.zip'
    img_to_download =  ['val','test','train']
    ann_to_download = ['annotations_trainval','image_info_test']
    base_url_images = 'http://images.cocodataset.org/zips/'
    base_url_ann = 'http://images.cocodataset.org/annotations/'


    click.echo(click.style(f"\n DOWNLOAD ANNOTATIONS \n", bg='green', bold=True, fg='white'))
    for ann in ann_to_download:

        ## build Urls
        ann_url = base_url_ann + ann + str(cfg.COCO_YEARS) + file_type
        
        click.echo(click.style(f'\nDownloading of {ann} ...\n', bg='blue', bold=True, fg='white'))
        click.echo(f'{ann} will be downloaded')

        zip_filename_location = save_zip_from_url(ann_url,cfg.PATH_ANNOTATIONS)
        #zip_filename_location = "/home/kamgo-gpu/Schreibtisch/stuff_annotations_trainval2017.zip"
        click.echo(f"the downloaded zip file was saved in to {zip_filename_location}")

        click.echo(click.style(f'\n Extraction of {ann} ...\n',  bg='blue', bold=True, fg='white'))
        click.echo(f'{ann} will be extracted and the zip-file will be deleted')

        # Extract zip to annotation directory
        Extract_zip_file(zip_filename_location,cfg.PATH_ANNOTATIONS)

    click.echo(click.style(f"\n DOWNLOAD IMAGES \n", bg='green', bold=True, fg='white'))
    for dataset in img_to_download:
        ## build Urls
        dataset_img_url = base_url_images + dataset + str(cfg.COCO_YEARS) + file_type
        
        click.echo(click.style(f'\n Downloading of {dataset} ...\n',  bg='blue', bold=True, fg='white'))
        click.echo(f'{dataset} will be downloaded')

        zip_filename_location = save_zip_from_url(dataset_img_url,cfg.PATH_IMAGES)
        click.echo(f"the downloaded zip file was saved in to {zip_filename_location}")
        click.echo(click.style(f'\n Extraction of {dataset} ...\n',  bg='blue', bold=True, fg='white'))
        click.echo(f'{dataset} will be extracted and the zip-File will be deleted')

        # set complet Path to save images
        Extract_zip_file(zip_filename_location,cfg.PATH_IMAGES)

    click.echo(click.style(f'\n Download and extraction termined successfull {dataset} ...\n',  bg='green', bold=True, fg='white'))
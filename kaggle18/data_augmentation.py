from skimage import transform
from param_config import *
import os
import numpy as np
import sys
import random
import warnings

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label



def data_aug(image,label,angel=30,resize_rate=0.9):
    flip = random.randint(0, 1)
    size = image.shape[0]
    rsize = random.randint(np.floor(resize_rate*size),size)
    w_s = random.randint(0,size - rsize)
    h_s = random.randint(0,size - rsize)
    sh = random.random()/2-0.25
    rotate_angel = random.random()/180*np.pi*angel
    print(flip,size,rsize,w_s,h_s,sh,rotate_angel)
    # Create Afine transform
    afine_tf = transform.AffineTransform(shear=sh,rotation=rotate_angel)
    # Apply transform to image data
    image = transform.warp(image, inverse_map=afine_tf,mode='edge')
    label = transform.warp(label, inverse_map=afine_tf,mode='edge')
    # Randomly corpping image frame
    image = image[w_s:w_s+size,h_s:h_s+size,:]
    label = label[w_s:w_s+size,h_s:h_s+size]
    # Ramdomly flip frame
    if flip:
        image = image[:,::-1,:]
        label = label[:,::-1]
    return image, label

def make_data_augmentation(image_ids,images,masks,split_num):
    for image_id,image, labels in zip(image_ids,images,masks):
        if not os.path.exists(TRAIN_PATH+"/{}/augs/".format(image_id)):
            os.makedirs(TRAIN_PATH+"/{}/augs/".format(image_id))
        if not os.path.exists(TRAIN_PATH+"/{}/augs_masks/".format(image_id)):
            os.makedirs(TRAIN_PATH+"/{}/augs_masks/".format(image_id))
        print(image_id,image.shape,labels.shape)   
        # also save the original image in augmented file 
        plt.imsave(fname=TRAIN_PATH+"/{}/augs/{}.png".format(image_id,image_id), arr = image)
        plt.imsave(fname=TRAIN_PATH+"/{}/augs_masks/{}.png".format(image_id,image_id),arr = np.squeeze(labels))

        for i in range(split_num):
            new_image, new_labels = data_aug(image,labels,angel=5,resize_rate=0.9)
            aug_img_dir = TRAIN_PATH+"/{}/augs/{}_{}.png".format(image_id,image_id,i)
            aug_mask_dir = TRAIN_PATH+"/{}/augs_masks/{}_{}.png".format(image_id,image_id,i)
            plt.imsave(fname=aug_img_dir, arr = new_image)
            plt.imsave(fname=aug_mask_dir,arr = np.squeeze(new_labels))

def clean_data_augmentation(image_ids):
    for ax_index, image_id in tqdm(enumerate(image_ids),total=len(image_ids)):
        if os.path.exists(TRAIN_PATH+"/{}/augs/".format(image_id)):
            shutil.rmtree(TRAIN_PATH+"/{}/augs/".format(image_id))
        if os.path.exists(TRAIN_PATH+"/{}/augs_masks/".format(image_id)):
            shutil.rmtree(TRAIN_PATH+"/{}/augs_masks/".format(image_id))

def get_augmented_data(train_ids,split_num):
    aug_num=split_num+1
    X_aug_train = np.zeros((len(train_ids)*aug_num, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    Y_aug_train = np.zeros((len(train_ids)*aug_num, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    print('Getting and resizing augmented train images and masks ... ')
    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        path = TRAIN_PATH + id_
        img = imread(path + '/augs/' + id_ +'.png')[:,:,:IMG_CHANNELS]
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X_aug_train[n*aug_num] = img
        mask = imread(path + '/augs_masks/' + id_ +'.png')[:,:,0]
        mask = np.expand_dims(resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', 
                                          preserve_range=True), axis=-1)
        mask[mask==1]=0
        mask[mask>0]=1
        Y_aug_train[n*aug_num] = mask
        for i in range(split_num):
            img = imread(path + '/augs/' + id_ +"_"+str(i)+'.png')[:,:,:IMG_CHANNELS]
            img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
            X_aug_train[n*aug_num+i] = img
            mask = imread(path + '/augs_masks/' + id_ +"_"+str(i)+'.png')[:,:,0]
            mask = np.expand_dims(resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', 
                                         preserve_range=True), axis=-1)
            mask[mask==1]=0
            mask[mask>0]=1
            Y_aug_train[n*aug_num+i] = mask
    return X_aug_train,Y_aug_train

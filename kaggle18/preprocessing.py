import sys
import random
import warnings
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label, binary_erosion, binary_dilation, disk
from skimage.morphology import square, watershed, closing, binary_closing
from skimage.morphology import remove_small_holes, remove_small_objects
from skimage.filters.rank import gradient
from skimage.exposure import rescale_intensity
from skimage.segmentation import random_walker

from sklearn.model_selection import KFold

from scipy.ndimage.morphology import binary_fill_holes

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.optimizers import Adam

import tensorflow as tf

import pickle as pkl
import gc

# Set some parameters
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
TRAIN_PATH = '/home/jma7/.kaggle/competitions/data-science-bowl-2018/stage1_train/'
TEST_PATH = '/home/jma7/.kaggle/competitions/data-science-bowl-2018/stage1_test/'
TEST2_PATH = '/home/jma7/.kaggle/competitions/data-science-bowl-2018/stage2_test/'

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed


train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]
test2_ids=next(os.walk(TEST2_PATH))[1]

X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
print('Getting and resizing train images and masks ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_train[n] = img
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    for mask_file in next(os.walk(path + '/masks/'))[2]:
        mask_ = imread(path + '/masks/' + mask_file)
        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', 
                                      preserve_range=True), axis=-1)
        mask = np.maximum(mask, mask_)
    Y_train[n] = mask


# Get and resize test images
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []
print('Getting and resizing test images ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[n] = img

print("stage1 loading done")

X2_test = np.zeros((len(test2_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test2 = []
print('Getting and resizing test images ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(test2_ids), total=len(test2_ids)):
    path = TEST2_PATH + id_
    try:
        img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    except:
        pass
    sizes_test2.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X2_test[n] = img


print('stage2 loading done!')


def make_data_augmentation(image_ids,images,masks,split_num):
    for image_id,image, labels in zip(image_ids,images,masks):
        
        path="/home/jma7/.kaggle/competitions/data-science-bowl-2018"
        if not os.path.exists(path+"/stage1_train/{}/augs/".format(image_id)):
            os.makedirs(path+"/stage1_train/{}/augs/".format(image_id))
        if not os.path.exists(path+"/stage1_train/{}/augs_masks/".format(image_id)):
            os.makedirs(path+"/stage1_train/{}/augs_masks/".format(image_id))
        print(image_id,image.shape,labels.shape)   
        # also save the original image in augmented file 
        plt.imsave(fname=path+"/stage1_train/{}/augs/{}.png".format(image_id,image_id), arr = image)
        plt.imsave(fname=path+"/stage1_train/{}/augs_masks/{}.png".format(image_id,image_id),arr = np.squeeze(labels))

        for i in range(split_num):
            new_image, new_labels = data_aug(image,labels,angel=5,resize_rate=0.9)
            aug_img_dir = path+"/stage1_train/{}/augs/{}_{}.png".format(image_id,image_id,i)
            aug_mask_dir = path+"stage1_train/{}/augs_masks/{}_{}.png".format(image_id,image_id,i)
            plt.imsave(fname=aug_img_dir, arr = new_image)
            plt.imsave(fname=aug_mask_dir,arr = np.squeeze(new_labels))

def clean_data_augmentation(image_ids):
    path="/hom3/jma7/.kaggle/competitions/data-science-bowl-2018"
    for ax_index, image_id in tqdm(enumerate(image_ids),total=len(image_ids)):
        if os.path.exists(path+"/stage1_train/{}/augs/".format(image_id)):
            shutil.rmtree(path+"/stage1_train/{}/augs/".format(image_id))
        if os.path.exists(path+"/stage1_train/{}/augs_masks/".format(image_id)):
            shutil.rmtree(path+"/stage1_train/{}/augs_masks/".format(image_id))

split_num = 10
make_data_augmentation(train_ids,X_train,Y_train,split_num)

'''
creat train and val training batch
'''
import os
import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg
import tensorflow.contrib.slim as slim

from random import sample
from collections import defaultdict
from tensorflow.contrib.slim.nets import inception



def prepare_batch(data_paths_and_classes,
                    batch_size):
    batch_paths_and_classes = sample(data_paths_and_classes, batch_size)
    images = [mpimg.imread(path) for path, labels 
            in batch_paths_and_classes]
    prepared_images = [image / 255 for image in images]
    X_batch = 2 * np.stack(prepared_images) - 1
    X_batch = np.expand_dims(X_batch,axis = 3)
    y_batch = np.array([labels for path, labels 
            in batch_paths_and_classes], dtype=np.int32)
    return X_batch, y_batch

def creat_batch(data_root_path, data_classes):
    data_classes_ids = {data_class:index for index, data_class in enumerate(data_classes)}                
    data_paths = defaultdict(list)
    for data_class in data_classes:
        image_dir = os.path.join(data_root_path, data_class)
        for filepath in os.listdir(image_dir):
            if filepath.endswith(".jpg"):
                data_paths[data_class].append(
                    os.path.join(image_dir, filepath))
    data_paths_and_classes = []
    for data_class, paths in data_paths.items():
        for path in paths:
            data_paths_and_classes.append((
                path, data_classes_ids[data_class]))
    return data_paths_and_classes
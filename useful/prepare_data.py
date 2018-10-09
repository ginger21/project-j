'''
divide pictiures into train and val tow parts
'''
import os
from random import sample
from collections import defaultdict
from PIL import Image
import numpy as np

def creat_paths(root_path, classes):
    image_paths = defaultdict(list)
    for class_ in classes:
        image_dir = os.path.join(root_path, class_)
        for filepath in os.listdir(image_dir):
            if filepath.endswith('.jpg'):
                image_paths[class_].append(os.path.join(image_dir, filepath))
    return image_paths


def prepare_data(image_paths, data_path, classes, ratio = 0.2):
    train_sizes = 0
    val_sizes = 0
    for class_ in classes:
        
        # 创建每类的文件夹
        train_path = os.path.join(data_path,'train', class_)
        if os.path.exists(train_path):
            if len(os.listdir(train_path)) > 0:
                train_file_num = int(os.listdir(train_path)[-1][0:5])
            else:
                train_file_num = 0
        else:    
            os.makedirs(train_path)
        
        val_path = os.path.join(data_path,'val', class_)
        if os.path.exists(val_path):
            if len(os.listdir(val_path)) > 0:
                val_file_num = int(os.listdir(val_path)[-1][0:5])
            else:
                val_file_num = 0
        else:
            os.makedirs(val_path)
        
        # 如果存在文件夹，其中文件数
    
        #train_file_num = int(os.listdir(train_path)[-1][0:5])
        #val_file_num = int(os.listdir(val_path)[-1][0:5])
        
        # train 数目
        train_size = int(len(image_paths[class_]) * (1 - ratio))
        val_size = len(image_paths[class_]) - train_size
        train_sizes += train_size
        val_sizes += val_size
        np.random.shuffle(image_paths[class_])
        train_files = image_paths[class_][:train_size]
        val_files = image_paths[class_][train_size:]

        # 生成训练数据
        for name, path in enumerate(train_files):
            pic_name = train_path + '/{:05d}.jpg'.format(train_file_num + name + 1)
            images = Image.open(path)
            images.save(pic_name)
        # 生成val数据
        for name, path in enumerate(val_files):
            pic_name = val_path + '/{:05d}.jpg'.format(val_file_num + name + 1)
            images = Image.open(path)
            images.save(pic_name)
        print(class_+' is done')
    return train_sizes, val_sizes        

classes = ['forward', 'left', 'right'
            ,'stop', 'turn_left', 'turn_right', 'walk']


dire_paths = creat_paths('c:/py/car_go/data/', classes)


prepare_data(dire_paths, 'c:/py/car_go', classes)
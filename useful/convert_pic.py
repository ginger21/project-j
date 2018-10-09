"""
convert csv data to picture and save 
"""
import os
import numpy as np
import pandas as pd
from PIL import Image

csv_data = 'C:/py/emo/fer2013/fer2013.csv'
tra_path_root = 'C:/py/emo/train'
val_path_root = 'C:/py/emo/val'

emo_data = pd.read_csv(csv_data)
emo_gb = dict(list(emo_data.groupby(['emotion', 'Usage'])))

# classes: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
classes={0:'Angry', 1:'Disgust', 2:'Fear', 5:'Surprise', 6:'Neutral'}


dataset_train = [[(emo_class_ids, 'Training'), 
               os.path.join(tra_path_root, classes[emo_class_ids])] 
                for emo_class_ids in classes]

dataset_val = [[(emo_class_ids, 'PublicTest'), 
               os.path.join(val_path_root, classes[emo_class_ids])] 
                for emo_class_ids in classes]


for classes, save_path in dataset_val:
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    pixel_ind = 0
    for pixels in emo_gb[classes]['pixels']:
        pixel = np.asarray([float(p) for p in pixels.split()]).reshape(48, 48)
        im = Image.fromarray(pixel).convert('L')
        image_name = os.path.join(save_path, '{:05d}.jpg'.format(pixel_ind))
        im.save(image_name)
        pixel_ind += 1
    print(classes, 'end')
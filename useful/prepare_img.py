'''
trans picture into target form
'''
import tensorflow as tf
from random import sample
import matplotlib.image as mpimg
from skimage.transform import resize
import numpy as np
import cv2


def prepare_image(image, target_width = 48, target_height = 48, max_zoom = 0.2):
    """Zooms and crops the image randomly for data augmentation."""

    # First, let's find the largest bounding box with the target size ratio that fits within the image
    height = image.shape[0]
    width = image.shape[1]
    image_ratio = width / height
    target_image_ratio = target_width / target_height
    crop_vertically = image_ratio < target_image_ratio
    crop_width = width if crop_vertically else int(height * target_image_ratio)
    crop_height = int(width / target_image_ratio) if crop_vertically else height
        
    # Now let's shrink this bounding box by a random factor (dividing the dimensions by a random number
    # between 1.0 and 1.0 + `max_zoom`.
    resize_factor = np.random.rand() * max_zoom + 1.0
    crop_width = int(crop_width / resize_factor)
    crop_height = int(crop_height / resize_factor)
    
    # Next, we can select a random location on the image for this bounding box.
    x0 = np.random.randint(0, width - crop_width)
    y0 = np.random.randint(0, height - crop_height)
    x1 = x0 + crop_width
    y1 = y0 + crop_height
    
    # Let's crop the image using the random bounding box we built.
    image = image[y0:y1, x0:x1]

    # Let's also flip the image horizontally with 50% probability:
    # if np.random.rand() < 0.5:
    #    image = np.fliplr(image)

    # Now, let's resize the image to the target dimensions.
    image = cv2.resize(image,(48,48),interpolation=cv2.INTER_CUBIC)
    image = image.astype(np.float32) / 255
    # Finally, let's ensure that the colors are represented as
    # 32-bit floats ranging from -1.0 to 1.0 (for now):
    return 2 * image - 1

pic_path = 'C:/py/emo/train/happy/00000.jpg'

example_image = mpimg.imread(pic_path)
test_image = prepare_image(example_image)
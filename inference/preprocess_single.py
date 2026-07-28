import numpy as np
import rasterio
import cv2
import torch


IMAGE_SIZE = (256, 256)


def normalize_db(img):

    img = np.nan_to_num(img) # to remove missing, nan

    img = np.clip(img, -35, 5) # makes sure that all pixel values are betwene -35 and 5

    img = (img + 35) / 40 # normalizing all pixel values in the range of 0 and 1 where -35 being 0 and 5 being 1

    return img.astype(np.float32)


def preprocess_image(path):

    with rasterio.open(path) as src: # safely closing files

        vv = src.read(1) # band 1

        vh = src.read(2) # band 2

    vv = cv2.resize(           # resizes image from 512 to 256
        vv,
        IMAGE_SIZE,
        interpolation=cv2.INTER_LINEAR  # produces smooth output
    )

    vh = cv2.resize(
        vh,
        IMAGE_SIZE,
        interpolation=cv2.INTER_LINEAR
    )

    vv = normalize_db(vv)

    vh = normalize_db(vh)

    image = np.stack([vv, vh], axis=0)  # channel * height * width 

    image = torch.tensor(   # convert array into tensor
        image,
        dtype=torch.float32
    )

    image = image.unsqueeze(0) # adds batch size dimension batch size * channel * height * width 

    return image

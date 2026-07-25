import numpy as np
import rasterio
import cv2
import torch


IMAGE_SIZE = (256, 256)


def normalize_db(img):

    img = np.nan_to_num(img)

    img = np.clip(img, -35, 5)

    img = (img + 35) / 40

    return img.astype(np.float32)


def preprocess_image(path):

    with rasterio.open(path) as src:

        vv = src.read(1)

        vh = src.read(2)

    vv = cv2.resize(
        vv,
        IMAGE_SIZE,
        interpolation=cv2.INTER_LINEAR
    )

    vh = cv2.resize(
        vh,
        IMAGE_SIZE,
        interpolation=cv2.INTER_LINEAR
    )

    vv = normalize_db(vv)

    vh = normalize_db(vh)

    image = np.stack([vv, vh], axis=0)

    image = torch.tensor(
        image,
        dtype=torch.float32
    )

    image = image.unsqueeze(0)

    return image
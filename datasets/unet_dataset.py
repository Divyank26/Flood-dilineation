import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class FloodSegmentationDataset(Dataset):

    def __init__(
        self,
        image_dir,
        mask_dir,
        labels_csv=None,
        flood_only=False
    ):

        self.image_dir = image_dir
        self.mask_dir = mask_dir

        # ---------------------------------
        # Load images
        # ---------------------------------

        if labels_csv is None:

            self.images = sorted(os.listdir(image_dir))

        else:

            df = pd.read_csv(labels_csv)

            if flood_only:

                df = df[df["label"] == 1]

            self.images = sorted(df["filename"].tolist())

        print("=" * 50)
        print("FloodSegmentationDataset")
        print("=" * 50)
        print(f"Images Loaded : {len(self.images)}")
        print(f"Flood Only    : {flood_only}")
        print("=" * 50)

    def __len__(self):

        return len(self.images)

    def __getitem__(self, idx):

        filename = self.images[idx]

        image = np.load(
            os.path.join(
                self.image_dir,
                filename
            )
        ).astype(np.float32)

        mask = np.load(
            os.path.join(
                self.mask_dir,
                filename
            )
        ).astype(np.float32)

        image = torch.from_numpy(image)

        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask
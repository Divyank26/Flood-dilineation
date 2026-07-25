import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class FloodClassifierDataset(Dataset):

    def __init__(self,
                 image_dir,
                 csv_file):

        self.image_dir = image_dir

        self.df = pd.read_csv(csv_file)

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        filename = row["filename"]

        label = row["label"]

        image = np.load(
            os.path.join(
                self.image_dir,
                filename
            )
        )

        image = torch.tensor(
            image,
            dtype=torch.float32
        )

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return image, label
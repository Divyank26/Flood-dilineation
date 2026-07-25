import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, random_split

from datasets.unet_dataset import FloodSegmentationDataset
from models.attention_unet import AttentionUNet
from losses.dice_loss import DiceLoss


def train():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print("Training Attention U-Net")
    print("=" * 60)

    print(f"Using device : {device}")

    if device.type == "cuda":

        print(
            f"GPU : {torch.cuda.get_device_name(0)}"
        )

    # -----------------------------------------
    # Dataset
    # -----------------------------------------

    dataset = FloodSegmentationDataset(
        image_dir="processed/images",
        mask_dir="processed/masks",
        labels_csv="processed/labels.csv",
        flood_only=False
    )

    train_size = int(0.8 * len(dataset))

    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0
    )

    print()

    print("Training Images :", len(train_dataset))

    print("Validation Images :", len(val_dataset))

    print()

    # -----------------------------------------
    # Model
    # -----------------------------------------

    model = AttentionUNet().to(device)

    bce = nn.BCEWithLogitsLoss()

    dice = DiceLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-4
    )

    epochs = 30

    best_loss = 999999

    # -----------------------------------------
    # Training
    # -----------------------------------------

    for epoch in range(epochs):

        start = time.time()

        model.train()

        train_loss = 0

        for images, masks in train_loader:

            images = images.to(device)

            masks = masks.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = bce(outputs, masks) + dice(outputs, masks)

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # -----------------------------

        model.eval()

        val_loss = 0

        with torch.no_grad():

            for images, masks in val_loader:

                images = images.to(device)

                masks = masks.to(device)

                outputs = model(images)

                loss = bce(outputs, masks) + dice(outputs, masks)

                val_loss += loss.item()

        val_loss /= len(val_loader)

        t = time.time() - start

        print(
            f"Epoch {epoch+1:02d}/30 "
            f"Train Loss {train_loss:.4f} "
            f"Val Loss {val_loss:.4f} "
            f"Time {t:.1f}s"
        )

        if val_loss < best_loss:

            best_loss = val_loss

            torch.save(
                model.state_dict(),
                "unet_best.pth"
            )

            print("Best model saved")

    print()

    print("=" * 60)

    print("Training Finished")

    print("Best Validation Loss :", best_loss)

    print("=" * 60)


if __name__ == "__main__":

    train()
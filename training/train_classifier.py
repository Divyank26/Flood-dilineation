import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from datasets.classifier_dataset import FloodClassifierDataset
from models.classifier import FloodClassifier


def train():

    # --------------------------------------------------
    # Device
    # --------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)

    print("Training Flood Classifier")

    print("=" * 60)

    print(f"Using device : {device}")

    if device.type == "cuda":

        print(
            f"GPU          : {torch.cuda.get_device_name(0)}"
        )

        print(
            f"CUDA Version : {torch.version.cuda}"
        )

    else:

        print("Running on CPU")

    print()

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    dataset = FloodClassifierDataset(
        image_dir="processed/images",
        csv_file="processed/labels.csv"
    )

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    # --------------------------------------------------
    # Dataset Statistics
    # --------------------------------------------------

    df = pd.read_csv("processed/labels.csv")

    num_flood = (df["label"] == 1).sum()
    num_no_flood = (df["label"] == 0).sum()

    print(f"Total Images : {len(df)}")
    print(f"Flood        : {num_flood}")
    print(f"No Flood     : {num_no_flood}")

    print()

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = FloodClassifier().to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    epochs = 20

    best_accuracy = 0

    # --------------------------------------------------
    # Training Loop
    # --------------------------------------------------

    for epoch in range(epochs):

        start = time.time()

        model.train()

        running_loss = 0

        correct = 0

        total = 0

        for images, labels in loader:

            images = images.to(device)

            labels = labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            predictions = (
                torch.sigmoid(outputs) > 0.5
            )

            correct += (
                predictions == labels.bool()
            ).sum().item()

            total += labels.size(0)

        accuracy = 100 * correct / total

        epoch_time = time.time() - start

        print(
            f"Epoch [{epoch+1:02d}/{epochs}] "
            f"Loss: {running_loss/len(loader):.4f} "
            f"Accuracy: {accuracy:.2f}% "
            f"Time: {epoch_time:.1f}s"
        )

        # ----------------------------------------------
        # Save Best Model
        # ----------------------------------------------

        if accuracy > best_accuracy:

            best_accuracy = accuracy

            torch.save(
                model.state_dict(),
                "classifier_best.pth"
            )

            print(
                f"Best model saved "
                f"({best_accuracy:.2f}%)"
            )

    print()

    print("=" * 60)

    print("Training Finished")

    print(f"Best Accuracy : {best_accuracy:.2f}%")

    print("Model saved as classifier_best.pth")

    print("=" * 60)


def main():

    train()


if __name__ == "__main__":

    main()
import torch
import torch.nn as nn
import torch.optim as optim

from datasets.classifier_dataset import FloodClassifierDataset
from models.classifier import FloodClassifier
from torch.utils.data import DataLoader

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = FloodClassifier().to(device)

model.load_state_dict(
    torch.load(
        "classifier_best.pth",
        map_location=device
    )
)

model.eval()

dataset = FloodClassifierDataset(
    "processed/images",
    "processed/labels.csv"
)

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=False
)

temperature = nn.Parameter(torch.ones(1, device=device))

optimizer = optim.LBFGS(
    [temperature],
    lr=0.01,
    max_iter=50
)

criterion = nn.BCEWithLogitsLoss()

logits_list = []
labels_list = []

with torch.no_grad():

    for images, labels in loader:

        images = images.to(device)
        labels = labels.float().to(device)

        logits = model(images)

        logits_list.append(logits)

        labels_list.append(labels)

logits = torch.cat(logits_list)

labels = torch.cat(labels_list)


def closure():

    optimizer.zero_grad()

    loss = criterion(
        logits / temperature,
        labels.unsqueeze(1)
    )

    loss.backward()

    return loss


optimizer.step(closure)

print("Optimal Temperature:", temperature.item())
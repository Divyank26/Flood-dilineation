import torch

from models.classifier import FloodClassifier


class FloodClassifierInference:

    def __init__(
        self,
        model_path,
        device
    ):

        self.device = device
        with open("classifier_temperature.txt") as f:
            self.temperature = float(f.read().strip())

        self.model = FloodClassifier().to(device)

        self.model.load_state_dict(
            torch.load(
                model_path,
                map_location=device,
                weights_only=True
            )
        )

        self.model.eval()

    def predict(self, image):

        with torch.no_grad():

            logits = self.model(image.to(self.device))

            probability = torch.sigmoid(
                logits / self.temperature
            ).item()

        return probability
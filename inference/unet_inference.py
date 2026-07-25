import torch

from models.attention_unet import AttentionUNet


class FloodSegmentationInference:

    def __init__(self, model_path, device):

        self.device = device

        self.model = AttentionUNet().to(device)

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

            output = self.model(
                image.to(self.device)
            )

            output = torch.sigmoid(output)

            mask = (output > 0.5).float()

        return mask.squeeze().cpu().numpy()
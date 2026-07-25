import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

import time
import torch
import cv2

from inference.preprocess_single import preprocess_image
from inference.classifier_inference import FloodClassifierInference
from inference.unet_inference import FloodSegmentationInference
from inference.statistics import FloodStatistics
from inference.visualization import create_overlay


class FloodPredictor:

    def __init__(self):

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.classifier = FloodClassifierInference(
            "classifier_best.pth",
            self.device
        )

        self.segmenter = FloodSegmentationInference(
            "unet_best.pth",
            self.device
        )

        self.stats = FloodStatistics()

    def predict(self, image_path):

        start = time.time()

        image = preprocess_image(image_path)

        probability = self.classifier.predict(image)

        result = {

            "classifier_probability": probability,

            "flood_detected": False,

            "mask": None,

            "overlay": None,

            "statistics": None,

            "inference_time": None

        }

        # Definitely No Flood
        if probability < 0.30:

            result["inference_time"] = (
                time.time() - start
            )

            return result

        # Segment

        mask = self.segmenter.predict(image)

        result["flood_detected"] = True

        result["mask"] = mask

        vv = image.squeeze()[0].numpy()

        overlay = create_overlay(vv, mask)

        result["overlay"] = overlay

        result["statistics"] = self.stats.compute(mask)

        result["inference_time"] = (
            time.time() - start
        )

        return result


if __name__ == "__main__":

    predictor = FloodPredictor()

    result = predictor.predict(
        "sen1floods11_data/S1/Bolivia_1009032_S1Weak.tif"
    )

    print()

    print("=" * 60)

    print(result)

    print("=" * 60)

    if result["overlay"] is not None:

        cv2.imwrite(
            "prediction_overlay.png",
            result["overlay"]
        )

    if result["mask"] is not None:

        cv2.imwrite(
            "prediction_mask.png",
            result["mask"] * 255
        )
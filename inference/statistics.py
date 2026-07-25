import numpy as np


class FloodStatistics:

    def __init__(self, pixel_area=100):
        """
        Sentinel-1 GRD
        10 m × 10 m
        = 100 m²
        """

        self.pixel_area = pixel_area

    def compute(self, mask):

        flood_pixels = int(mask.sum())

        total_pixels = mask.size

        flood_percentage = (
            flood_pixels / total_pixels
        ) * 100

        area_m2 = flood_pixels * self.pixel_area

        area_km2 = area_m2 / 1e6

        return {

            "flood_pixels": flood_pixels,

            "total_pixels": total_pixels,

            "flood_percentage": flood_percentage,

            "area_m2": area_m2,

            "area_km2": area_km2

        }
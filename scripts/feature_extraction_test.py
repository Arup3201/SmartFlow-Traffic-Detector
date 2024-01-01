from pathlib import Path
import time
from feature_extraction.hog_feature_extractor import HistogramOrientedGradients

if __name__=="__main__":

    image_paths = ['./data/test_images/bus-1.jpg', './data/test_images/bus-2.jpg', './data/test_images/car-1.jpg', './data/test_images/car-2.jpg', './data/test_images/truck-1.jpg', './data/test_images/truck-2.jpg']

    HOG = HistogramOrientedGradients()

    print("Starting feature extraction...")

    for img_path in image_paths:
        start_time = time.time()

        img_path = Path(img_path)
        features = HOG.hog(img_path)
        print(f"Feature shape: {features.shape}")

        end_time = time.time()

        print(f"Time taken to perform feature extraction operation: {end_time-start_time}")

from pathlib import Path
import time
from feature_extraction.hog_feature_extractor import HistogramOrientedGradients
from skimage.feature import hog
from PIL import Image

if __name__=="__main__":

    image_path = './data/test_images/bus-2.jpg'

    HOG = HistogramOrientedGradients()

    img_path = Path(image_path)
    features = HOG.hog(img_path)
    print(f"Feature shape: {features.shape}")


    start_time = time.time()
    features = HOG.hog(img_path)
    end_time = time.time()

    print(f"Time taken by custom hog method: {end_time-start_time}")

    start_time = time.time()
    img = Image.open(image_path)
    img = img.resize((128, 64))
    features = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False, channel_axis=-1)
    end_time = time.time()
    print(f"Time taken by skimage hog method: {end_time - start_time}")
    print(f"Feature shape: {features.shape}")
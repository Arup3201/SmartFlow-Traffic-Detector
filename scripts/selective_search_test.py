from region_proposals.selective_search import show_regions
from pathlib import Path

img_path = Path('./data/test_images/car-1.jpg')

show_regions(img_path=img_path, method='f')
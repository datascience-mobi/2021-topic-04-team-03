import pathlib as pl
import numpy as np
from skimage.io import imread_collection
from nuclei_segmentation import preprocessing, otsu, evaluation
import json

with open('../Results/new_values.json', 'r') as values_file:
    data = json.load(values_file)


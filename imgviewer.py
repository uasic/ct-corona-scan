import sys
import os

import numpy as np
from matplotlib import pyplot as plt

def prikazi_sliko(file_path):
    if (file_path.endswith(".npy")):
        img_array = np.load(file_path)

        img_array = img_array.squeeze()

        plt.imshow(img_array, cmap='gray')
        plt.show()

file_path = "" #"segmentations/processed/test/0004_slice0.npy"

if ( len(sys.argv) > 1 ):
    file_path = sys.argv[1]
else:
    raise NameError("Ni parametra")

print(file_path)

if (file_path.endswith(".npy")):
    prikazi_sliko(file_path)
else:
    stevilka = os.path.basename(file_path)
    dir = os.path.dirname(file_path)
    for name in os.listdir(dir):
        if (name.startswith(stevilka)):
            ime = os.path.join(dir, name)
            prikazi_sliko(ime)
    
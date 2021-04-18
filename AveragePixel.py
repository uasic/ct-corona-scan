# importing libraries
import cv2
import numpy as np
from skimage import io
import os
from collections import defaultdict
import matplotlib.pyplot as plt

def naivno_povprecje(img):
    return np.mean(img)
    
def optimizirano_povprecje(img):
    # 2D matriko v 1D vektor
    img_1d = img.flatten()
    img_1d = img_1d[img_1d != 0]
    return np.mean(img_1d)
    
def white_area_ration(img):
    vse_kar_ni_crno = img[img != 0]
    zelo_belo = vse_kar_ni_crno[np.where(vse_kar_ni_crno > 0.90)]
    #print(zelo_belo.size, vse_kar_ni_crno.size)
    return zelo_belo.size / vse_kar_ni_crno.size

# reading the image data from desired directory
# img = np.load("segmentations/processed/train/0484_slice5.npy")

# za vse datoteke v mapi C:\Users\urska\Desktop\HACKATHON\segmentations\processed\train moraš izvesti program

"""
povprecja = {
    0440: [0.20438269, ...],
    ...
}
"""

povprecja = defaultdict(list)

directory = 'segmentations/processed/train'

counter = 0

for filename in os.listdir(directory):
    if filename.endswith(".npy"):
        id = filename.split("_")[0]
        img = np.load(os.path.join(directory, filename))
        povprecje = white_area_ration(img)
        povprecja[id].append(povprecje)
        counter -= 1
        if (counter == 0):
            break

print(povprecja)

avg_of_avgs = {k: np.mean(l) for (k, l) in povprecja.items()} # mean/max

print(avg_of_avgs)


classification_file_path = "train.txt"

lines = []
with open(classification_file_path) as f:
    lines = [line.rstrip() for line in f]

razredi = {}
for line in lines:
    tmp = line.split(",")
    razred = tmp[1]
    key = tmp[0].split(".")[0]
    razredi[key] = razred

#print(razredi)

#
"""
enke = [k for (k, _) in razredi.items()]
enke_vrednosti = {k: avg_of_avgs[k] for k in enke if k in avg_of_avgs}
urejeno = dict(sorted(enke_vrednosti.items(), key=lambda item: item[1]))
#print("\n", urejeno, "\n")
"""
#
fig, ax = plt.subplots()

x = []
y = []
for (key, avg) in avg_of_avgs.items():
    x.append(avg)
    y.append(razredi[key])
    ax.annotate(key, (avg, razredi[key]))

plt.plot(x, y, 'o')

plt.show()
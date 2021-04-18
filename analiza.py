import cv2
import numpy as np
from skimage import io
import os
from collections import defaultdict
import matplotlib.pyplot as plt

from obmocja import najdi_poskodovana_obmocja, np_to_cv_img
from sklearn.linear_model import LogisticRegression
from permutation_test import permutation_test

DIR_PATH = 'segmentations/processed/train'

# "./../0440_slice0.npy" -> "0440"
def pridobi_id(pot_datoteke):
    return os.path.basename(pot_datoteke).split("_")[0]
    
"""
{
    0440: 1,
    0349: 0,
}
"""
def pridobi_pripadajoce_razrede():
    CLASSIFICATION_FILE_PATH = "train.txt"
    lines = []
    with open(CLASSIFICATION_FILE_PATH) as f:
        lines = [line.rstrip() for line in f]

    razredi = {}
    for line in lines:
        tmp = line.split(",")
        razred = tmp[1]
        key = tmp[0].split(".")[0]
        razredi[key] = int(razred)
    return razredi
        
CLASSES = pridobi_pripadajoce_razrede()
        
def narisi(vrednosti, razredi = CLASSES, ANNOTATE = False):
    fig, ax = plt.subplots()
    x = []
    y = []
    for (key, avg) in vrednosti.items():
        x.append(avg)
        y.append(razredi[key])
        if (ANNOTATE):
            ax.annotate(key, (avg, razredi[key]))

    plt.plot(x, y, 'o')
    plt.show()
    
"""
{
    0440: 1.32323,
    0349: 4.32343,
}
"""
def analiziraj(vrednotna_funkcija, agregacijska_funkcija = np.mean, directory = DIR_PATH, plot = False):
    vrednosti = defaultdict(list)
    
    if vrednotna_funkcija is None:
        raise ValueError("vrednotna_funkcija is not set")

    for filename in os.listdir(directory):
        if filename.endswith(".npy"):
            np_img = np.load(os.path.join(directory, filename))
            vrednost = vrednotna_funkcija(np_img)
            vrednosti[pridobi_id(filename)].append(vrednost)
            
    # funkcija, ki sprejme vektor in vrne skalar
    agregacije = {k: agregacijska_funkcija(l) for (k, l) in vrednosti.items()}
    
    if (plot is True):
        narisi(agregacije, ANNOTATE = False)
        
    return agregacije
    
def dataset():
    print("Extracting data from images to build dataset ...")
    
    def skupna_povrsina_poskodb(np_img):
        contours, indices, areas = najdi_poskodovana_obmocja(np_to_cv_img(np_img))
        return sum(areas)
        
    def stevilo_poskodb(np_img):
        contours, indices, areas = najdi_poskodovana_obmocja(np_to_cv_img(np_img))
        return len(indices)


    def koorduinata_najvecje_poskodbe(katera_koordinata):
        assert katera_koordinata == 0 or katera_koordinata == 1
        def x_koorduinata_najvecje_poskodbe(np_img):
            contours, indices, areas = najdi_poskodovana_obmocja(np_to_cv_img(np_img))
            ix_max = -1
            area_max = float('-inf')
            for ix, area in enumerate(areas):
                if (area > area_max):
                    area_max = area
                    ix_max = ix

            ix_in_contours = indices[ix_max]

            cnt_with_max_area = contours[ix_in_contours]

            return np.mean([coor[0][katera_koordinata] for coor in cnt_with_max_area])
        return x_koorduinata_najvecje_poskodbe

        
    def get(x):
        def f(l):
            return l[x]
        return f
    
        
    features_funcs = [
        (x_koorduinata_najvecje_poskodbe, np.mean),
        (skupna_povrsina_poskodb, max),
        (stevilo_poskodb, np.mean),
    ]        
    
    data = [ analiziraj(f, a) for f, a in features_funcs ]
    y =  np.array([ CLASSES[k] for k in data[0].keys() ])
    X = np.matrix([ list(d.values()) for d in data ]).T
    
    print("Dataset built.")
    
    assert len(X) == len(y)
    return X, y

if (__name__ == "__main__"):
    
    """
    [
        [1.32323, 3434.3434],
        [4.32343, 434.5634],
    ]
    """
    
    
    """
    [1, 0]
    """
    
    X, y = dataset()
    
    permutation_test(LogisticRegression(random_state=0).fit, X, y, iterations=10)
    exit()
    
    print("predict", clf.predict(X[:2, :]))
    
    print("predict_proba", clf.predict_proba(X[:2, :]))
    
    print("Score:", clf.score(X, y))
    
    pass
    
    
    
import sys
import numpy as np
import nrrd
import nibabel as nib
import os
import pickle
from cv2 import resize

def normalize(image, MIN_BOUND, MAX_BOUND):
    """
    This function normalizes CT image inside bounds MIN_BOUND, MAX_BOUND. 
    """
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image


def bound_box_and_reshape(img, slice_idx):
    """
    Crop given slice of image to lung size (remove redundant empty space) and reshape to 512x512 pxls. Return edited img_slice.
    """
    img_slice = img[:,:,slice_idx]
    rows = np.any(img_slice, axis=1)
    cols = np.any(img_slice, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    img_slice = resize(img_slice[rmin:rmax, cmin:cmax], (512,512))
    img_slice = np.transpose(img_slice[:, :, np.newaxis], axes = [2, 0, 1]).astype('float32')
    
    return img_slice


def mask_original(filepath_CT, filepath_mask, nb_central_slices=10):
    """
    Mask and normalize original CT. Select only preset number of central slices. 
    """
    I = np.array(nib.load(filepath_CT).dataobj)
    I = normalize(I, -1350, 150)
    M, _ = nrrd.read(filepath_mask)
    
    nS = np.where(M==1, I, M)
    
    z = nS.shape[2]//2
    dz = nb_central_slices//2
    nS = nS[:,:,z-dz:z+dz]
        
    return nS


if __name__ == "__main__":

    mode = "test"

    if ( len(sys.argv) > 1 ):
        mode = sys.argv[1]
        
    if (mode != "test" and mode != "train"):
        raise NotImplementedError("mode {mode} is not implemented!".format(mode))
    
    CTs_path = os.path.join("images/", mode)
    masks_path = os.path.join("segmentations/", mode)
    output_path = os.path.join("segmentations/processed/", mode)
    
    for name_CT in os.listdir(CTs_path):
        if name_CT.endswith(".gz"):
            name_mask = name_CT.replace(".nii.gz", ".nrrd")

            filepath_CT = os.path.join(CTs_path, name_CT)
            filepath_mask = os.path.join(masks_path, name_mask)
            
            img = mask_original(filepath_CT, filepath_mask)
            
            n_slices = 10
            print(name_CT)
            for slice_idx in range(n_slices):
                try:
                    img_slice = bound_box_and_reshape(img, slice_idx)
                except: 
                    raise NameError('Bounding and reshaping failed.')

                slice_name = name_CT.strip('.nii.gz') + "_slice" + str(slice_idx) + '.npy'
                np.save(os.path.join(output_path, slice_name), img_slice)


"""
CTs_path = "images/train"
masks_path = "segmentations/train"
output_path = "segmentations/processed/train/"
name_CT = "0443.nii.gz"
        
if name_CT.endswith(".gz"):
    name_mask = name_CT.replace(".nii.gz", ".nrrd")

    filepath_CT = os.path.join(CTs_path, name_CT)
    filepath_mask = os.path.join(masks_path, name_mask)
    
    img = mask_original(filepath_CT, filepath_mask)
    
    n_slices = 10
    print(name_CT)
    for slice_idx in range(n_slices):
        try:
            img_slice = bound_box_and_reshape(img, slice_idx)
        except: 
            raise NameError('Bounding and reshaping failed.')

        slice_name = name_CT.strip('.nii.gz') + "_slice" + str(slice_idx) + '.npy'
        np.save(os.path.join(output_path, slice_name), img_slice)
"""
                    

    
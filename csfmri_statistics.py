#!/usr/bin/env python

# DESCRIPTION

"""This Python module is intended for statistical comparison and evaluation of
results."""

# IMPORTS

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp
from glob import glob
import os
from time import time
from sklearn.cluster import KMeans


# DEFINITIONS AND CODE

def compare_maps(map1, map2, mask=None):
    """Compares the cardiac and respiratory maps. Implements a one-sample t-test
    for the pairwise difference of voxel values."""

    print "Loading volumes..."
    if mask:
        mask = nib.load(mask).get_data()
        mask[mask != 0] = 1.0
        map1 = nib.load(map1).get_data() * mask
        map2 = nib.load(map2).get_data() * mask
    else:
        map1 = nib.load(map1).get_data()
        map2 = nib.load(map2).get_data()
    map1 = map1 - np.mean(map1)
    map2 = map2 - np.mean(map2)
    diff = map2 - map1
    print "Running t-test..."
    t, p = ttest_1samp(diff.ravel(), 0)
    print "p-value:", p


def tSNR(img):
    """Calculates the tSNR map for a time series (4D) NIfTI image."""

    # Validate dimensions
    _img = np.array(img).astype(np.float32)
    assert len(_img.shape) == 4,\
        "The time series image must have four dimensions."

    # Exclude invalid voxel values
    _img[~np.isfinite(_img)] = np.nan
    invalid_voxel_count = np.flatnonzero(~np.isfinite(_img)).size > 0
    if invalid_voxel_count:
        print ("WARNING: There are {} invalid voxels"
               .format(invalid_voxel_count))

    # Calculate tSNR and set invalid values to zero
    tsnr = np.nanmean(_img, axis=-1) / np.nanstd(_img, axis=-1)
    tsnr[~np.isfinite(tsnr)] = 0

    return tsnr


def tsnr_wrapper(filepath, outputdir="output", n_echos=1, min_vols=100):
    """Calculates tSNR for a group of MRI images."""

    mrifiles = glob(filepath)

    # Create output directory if it does not exist
    if not os.path.isdir(outputdir):
        try:
            os.makedirs(outputdir)
        except:
            print ("ERROR: output directory could not be created.")
            exit(1)

    for mrifile in mrifiles:

        # Update status
        print ("Processing {}...".format(mrifile))

        # Load image
        mri = nib.load(mrifile)
        img = mri.get_data()
        hdr = mri.header

        # Calculate the tSNR map
        # Bear in mind that multiple echos are concatenated in the NIfTI format!
        tsnr_maps = []
        try:
            assert len(img.shape) == 4, "The input must be a 4D image."
            if (img.shape[3] < min_vols) or (img.shape[3] % n_echos != 0):
                print ("SKIPPED: Unexpected number of volumes.")
                continue

            means = np.mean(img, axis=(0,1,2))
            if all([np.all(means[::n_echos] - means[i::n_echos] > 0)
                    for i in range(1, n_echos)]):
                print ("Identified as multi-echo image.")
                for i in range(n_echos):
                    tsnr_maps.append(tSNR(img[:,:,:,i::n_echos]))
            else:
                print ("Identified as single-echo image.")
                tsnr_maps.append(tSNR(img))
        except AssertionError as exc:
            print ("SKIPPED: " + exc.message)
            continue

        # Adjust header and save the tSNR map
        hdr.set_data_shape(img.shape[:-1])
        hdr.set_data_dtype(np.float32)
        try:
            for i, tsnr_map in enumerate(tsnr_maps):
                fout = os.path.split(mrifile)[-1]\
                    .replace(".nii.gz", "_echo{:d}_tSNR.nii.gz".format(i))
                nib.save(nib.Nifti1Image(tsnr_map, hdr.get_sform(), hdr),
                         os.path.join(outputdir, fout))
        except:
            print ("ERROR: tSNR map could not be saved.")
            continue


def analyse_cardmap(imfile):
    """This is a collection of analyses for a cardiac map."""

    # Load and check shape
    try:
        img = nib.load().get_data()
    except:
        print ("The cardiac map could not be loaded.")
        exit(1)
    assert len(img.shape) == 3, "The input must be a 3D image."

    # Count all voxels
    print ("Total number of voxels: {}".format(np.prod(img.shape)))

    # Count invalid voxels
    invalid_voxel_coords = np.where(~np.isfinite(img))
    print ("Number of invalid voxels: {}".format(invalid_voxel_coords[0].size))

    # Count significant voxels
    print ("Number of significantly pulsatile voxels: {}"
           .format(np.count_nonzero(img[np.isfinite(img)])))


if __name__ == "__main__":
    """
    # Check if two maps are identical using unpaired t-test.
    compare_maps(map1="/Users/inhuszar/csfmri/analysis_python/results_to_keep/"
                 "F3T_2013_40_363_cardmap.nii.gz",
                 map2="/Users/inhuszar/csfmri/analysis_matlab/F3T_2013_40_363/"
                 "CardiacMap.nii.gz",
                 mask="/Users/inhuszar/csfmri/analysis_python/results_to_keep/"
                 "T12func_bin.nii.gz")

    compare_maps(map1="/Users/inhuszar/csfmri/analysis_python/results_to_keep/"
                      "F3T_2013_40_363_respmap.nii.gz",
                 map2="/Users/inhuszar/csfmri/analysis_matlab/F3T_2013_40_363/"
                      "RespMap.nii.gz",
                 mask="/Users/inhuszar/csfmri/analysis_python/results_to_keep/"
                      "T12func_bin.nii.gz")
    
    """
    # Calculate tSNR maps for all images of a subject
    start_t = time()
    tsnr_wrapper(
        filepath="/Volumes/INH_1TB/CSFMRI/data/F3T_2013_40_363/*.nii.gz",
        outputdir="/Volumes/INH_1TB/CSFMRI/analysis_python/F3T_2013_40_363/"
                  "tSNR", n_echos=3, min_vols=500)
    print ("Elapsed time: {} s.".format(time() - start_t))
    start_t = time()
    tsnr_wrapper(
        filepath="/Volumes/INH_1TB/CSFMRI/data/F3T_2013_40_370/*.nii.gz",
        outputdir="/Volumes/INH_1TB/CSFMRI/analysis_python/F3T_2013_40_370/"
                  "tSNR", n_echos=3, min_vols=500)
    print ("Elapsed time: {} s.".format(time() - start_t))
    start_t = time()
    tsnr_wrapper(
        filepath="/Volumes/INH_1TB/CSFMRI/data/F3T_2013_40_372/*.nii.gz",
        outputdir="/Volumes/INH_1TB/CSFMRI/analysis_python/F3T_2013_40_372/"
                  "tSNR", n_echos=3, min_vols=500)
    print ("Elapsed time: {} s.".format(time() - start_t))

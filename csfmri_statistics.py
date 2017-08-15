#!/usr/bin/env python

# DESCRIPTION

"""This Python module is intended for statistical comparison and evaluation of
results."""

# IMPORTS

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp


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


if __name__ == "__main__":
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
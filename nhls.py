#!/usr/bin/env python

# IMPORTS

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import sys
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.filters import sobel
from itertools import product
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage.filters import convolve
from time import time


# DESCRIPTION

usermanual = \
    """This utility tries to segment a 3D TOF (time-of-flight) MR image to 
create an MR Angiogram (MRA)."""

# This module implements the algorithm originally described in:
# Jiaxin Wang, Shifeng Zhao, Zifeng Liu, Yun Tian, Fuqing Duan, and Yutong Pan,
# "An Active Contour Model Based on Adaptive Threshold for Extraction of
# Cerebral Vascular Structures", Computational and Mathematical Methods in
# Medicine, vol. 2016, Article ID 6472397, 9 pages, 2016.
# doi:10.1155/2016/6472397


# DEFINITIONS AND CODE

# Width of the regularised Dirac-delta function
EPSILON = 1.0

# Vesselness shape descriptor coefficients
V_ALPHA = None
V_BETA = None
V_GAMMA = None

# Coefficient in [0.5, 1] for locally-specified dynamic threshold computation
K = 1

# Gaussian convolution kernel parameters
KERNEL_SIGMA = 1.944
KERNEL_RADIUS = int(round(3 * KERNEL_SIGMA))

# Energy function coefficients
ALPHA1 = 0.003
ALPHA2 = 0.003
BETA = 0.02
GAMMA = 1.0
MU_0 = 100

# Time increment
DT = 2.0

# Convergence threshold
PERCENT_CONVERGENCE = 1.0


def _h(x, epsilon=EPSILON):
    """Smooth Heaviside function."""
    return 0.5 + np.arctan(x / epsilon) / np.pi


def _delta(x, epsilon=EPSILON):
    """Dirac delta function"""
    return (epsilon / (epsilon ** 2 + x ** 2)) / np.pi


def _div(field):
    """Calculates the divergence of a vector field."""
    return np.sum(np.stack([np.gradient(field[..., i])[i]
                            for i in range(field.shape[-1])],
                           axis=field.ndim - 1), axis=-1)


# Modified from source: https://stackoverflow.com/questions/31206443/
# numpy-second-derivative-of-a-ndimensional-array
def _hessian(x):
    """
    Calculate the hessian matrix with finite differences
    Parameters:
       - x : ndarray
    Returns:
       an array of shape x.shape + (x.ndim, x.ndim)
       where the array[... i, j] corresponds to the second derivative x_ij
    """
    x_grad = np.gradient(x)
    hessian = np.empty(x.shape + (x.ndim, x.ndim), dtype=x.dtype)
    for k, grad_k in enumerate(x_grad):
        # iterate over dimensions
        # apply gradient again to every component of the first derivative.
        tmp_grad = np.gradient(grad_k)
        for l, grad_kl in enumerate(tmp_grad):
            hessian[..., k, l] = grad_kl
    return hessian


def _laplacian(field):
    """Calculates the Laplacian of an n-dimensional scalar field."""
    return _div(np.stack(np.gradient(field), axis=field.ndim))


def _vesselness(Ra, Rb, S, eigvals, alpha=None, beta=None, gamma=None):
    """Calculates vesselness score based on indicators of structuredness."""

    # These parameter settings looked intuitive to me, albeit they have not been
    # mentioned in the literature
    if alpha is None:
        alpha = np.std(Ra[np.nonzero(Ra)])
    if beta is None:
        beta = np.std(Rb[np.nonzero(Rb)])
    if gamma is None:
        gamma = np.std(S[np.nonzero(S)])

    res = np.zeros_like(Rb)
    roi = np.where(np.logical_and(eigvals[..., 1] <= 0,
                                  eigvals[..., 2] <= 0))
    res[roi] = (1.0 - np.exp(-Ra[roi] / (2 * alpha ** 2))) * \
               np.exp(-Rb[roi] ** 2 / (2 * beta ** 2)) \
               * (1.0 - np.exp(-S[roi] ** 2 / (2 * gamma ** 2)))
    return res


def _shift(img, dirs, fill_value=0):
    """Shifts an N-D image with the specified extent along each dimension.
    Linear interpolation is used to translate the image. The output has the same
    size and shape as the input."""

    _dirs = np.asarray(dirs)
    assert img.ndim == _dirs.size, \
        "The inputs must have identical dimensionality."

    # Set up interpolator
    axes = tuple(np.arange(0, i) for i in img.shape)
    ipol = RegularGridInterpolator(axes, img, bounds_error=False,
                                   fill_value=fill_value, method='linear')

    # Calculate new coordinates
    new_axes = []
    for k in range(_dirs.size):
        new_axes.append(np.asarray(axes[k]) - dirs[k])

    # Return shifted image
    return ipol(np.stack(np.meshgrid(*tuple(new_axes), indexing='ij'))
                .reshape(_dirs.size, -1).T).reshape(img.shape)


# The implementation of the N-D Canny edge detector was based on the following
# description:
# http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/
# py_imgproc/py_canny/py_canny.html

def cannyND(img, sigma=1, minval=None, maxval=None):
    """Canny edge detection for N dimensional images."""

    dim = img.ndim

    # Gaussian filtering
    _img = gaussian_filter(img, sigma=sigma)

    # Sobel filtering in all 3 directions
    sfimg = np.stack([sobel(_img, axis=i) for i in range(dim)], axis=dim)
    magnitudes = np.linalg.norm(sfimg, axis=-1)

    # Find local maxima of the gradient magnitude
    # pdirs: principal directions (neighbourhood in N dimensions)
    pdirs = np.stack(product(*((-1, 0, 1),) * dim))
    pdirs = pdirs[np.any(pdirs != 0, axis=-1)]
    pdirs = pdirs[:pdirs.shape[0]/2, :]
    nbix = np.argmax(np.abs(np.sum(sfimg[..., np.newaxis, :] * pdirs, axis=-1)
                     / np.linalg.norm(pdirs, axis=-1)
                     / np.repeat(magnitudes[..., np.newaxis], pdirs.shape[0],
                                 axis=-1)), axis=dim)
    edges = np.zeros_like(magnitudes)
    for k, direction in enumerate(pdirs):
        current_voxels = magnitudes[np.where(nbix == k)]
        ref1 = _shift(magnitudes, direction)[np.where(nbix == k)]
        ref2 = _shift(magnitudes, -direction)[np.where(nbix == k)]
        edges[np.where(nbix == k)] = \
            np.logical_and(current_voxels > ref1, current_voxels > ref2)\
                .astype(np.int8)
        # Release memory
        del current_voxels
        del ref1
        del ref2
    magnitudes *= edges

    # Set default values of minval and maxval
    if minval is None:
        minval = np.percentile(magnitudes, 50)
        print ("Minval: {}".format(minval))
    if maxval is None:
        maxval = np.percentile(magnitudes, 95)
        print ("Maxval: {}".format(maxval))

    # Handle user error
    if maxval < minval:
        print ("WARNING: minval < maxval. Automatic correction: "
               "minval = maxval.")
        minval = maxval

    # Histeresis thresholding
    edges = np.where(magnitudes > minval, 1, 0)
    edges_certain = np.where(magnitudes > maxval, 1, 0)
    nb_exploration = np.zeros(edges.shape + (pdirs.shape[0],))
    for k, direction in enumerate(pdirs):
        nb_exploration[..., k] = _shift(edges_certain, direction)
    edges *= np.any(nb_exploration, axis=-1)

    return edges


def _kernel(img, sigma=KERNEL_SIGMA, radius=KERNEL_RADIUS):
    """Localised Gaussian convolution kernel."""

    dim = img.ndim
    if radius is not None:
        kernel = \
            np.stack(np.meshgrid(*(np.arange(2 * radius + 1),) * dim,
                                 indexing='ij'), axis=dim).astype(np.float64)
    else:
        kernel = np.stack(np.meshgrid(*tuple(np.arange(i) for i in img.shape)),
                          axis=dim).astype(np.float64)
    kernel -= np.array(kernel.shape[:-1]) / 2.0
    kernel = np.linalg.norm(kernel, axis=-1)
    kernel = np.exp(-kernel / (2 * sigma ** 2)) \
             / np.sqrt(2 * np.pi * sigma ** 2)

    return kernel


def _g(x):
    """Regularised gradient function"""
    return np.divide(1.0, (1.0 + x ** 2))


def _grad(field):
    """Gradient of a scalar field"""
    return np.stack(np.gradient(field), axis=field.ndim)


def acm(tofimg):
    """
    :param ndarray tofimg: 3D bias-corrected TOF (time-of-flight) image.
    """

    # 1. Initialise vessel locations and their approximate boundaries
    # (Frangi's vessel enhancement algorithm)

    # 1.1 Obtain the Hessian matrix for all voxels, perform eigenvalue
    # decomposition and order the eigenpairs by the magnitude of the eigenvalues
    # (the order is ascending)
    eigvals, eigvects = np.linalg.eig(_hessian(tofimg))
    eigval_order = np.argsort(np.abs(eigvals), axis=-1)
    grids = np.ogrid[[slice(0, i) for i in eigvals.shape]]
    eigvals = eigvals[tuple(grids)[:-1] + (eigval_order,)]
    grids = np.ogrid[[slice(0, i) for i in eigvects.shape]]
    eigvects = eigvects[tuple(grids)[:-1]
                        + (np.expand_dims(eigval_order, axis=tofimg.ndim),)]

    # 1.2 Define shape descriptors
    Ra = np.abs(eigvals[..., 1].astype(np.float64)) \
         / np.abs(eigvals[..., 2].astype(np.float64))
    Ra[~np.isfinite(Ra)] = 0
    Rb = np.abs(eigvals[..., 0].astype(np.float64)) \
         / np.sqrt(np.abs(eigvals[..., 1].astype(np.float64))
                   * np.abs(eigvals[..., 2].astype(np.float64)))
    Rb[~np.isfinite(Rb)] = 0
    S = np.linalg.norm(eigvals.astype(np.float64), axis=-1)

    # 1.3 Calculate vesselness score
    R = _vesselness(Ra, Rb, S, eigvals, alpha=V_ALPHA, beta=V_BETA,
                    gamma=V_GAMMA)

    # 1.4 Run Canny edge detection to initialise the contour
    phi = 1.0 - cannyND(tofimg, sigma=1)


    # 2. Run active contour segmentation
    # 2.1 Calculate kernel function
    kernel = _kernel(tofimg, sigma=KERNEL_SIGMA, radius=KERNEL_RADIUS)

    iteration = 0
    while True:

        # Update status
        start_t = time()
        iteration += 1
        print ("Starting iteration No. {}...".format(iteration))

        # 2.2 Calculate locally-specified dynamic threshold
        phi_h = _h(phi, epsilon=EPSILON)
        mu = K * convolve(phi_h * tofimg, kernel) / convolve(phi_h, kernel)

        # 2.3 Calculate penalty term
        #P = np.sum(0.5 * np.linalg.norm(grad_phi - 1, axis=-1) ** 2)
        grad_phi = _grad(phi)
        P = _laplacian(phi) - \
            _div(np.divide(grad_phi, np.repeat(np.linalg.norm(grad_phi, axis=-1)
                                               [..., np.newaxis],
                                               grad_phi.shape[-1], axis=-1)))
        P[~np.isfinite(P)] = 0

        # 2.4 Calculate the terms of the temporal derivative of the contour
        M1 = tofimg - MU_0
        M2 = tofimg - mu
        grad_phi = _grad(phi)
        N = _div(_g(np.divide(grad_phi,
                              np.repeat(np.linalg.norm(grad_phi, axis=-1)
                                        [..., np.newaxis], grad_phi.shape[-1],
                                        axis=-1))))
        N[~np.isfinite(N)] = 0

        # 2.5 Update the contour
        phi += DT * (ALPHA1 * M1 + ALPHA2 * M2 + BETA * N + GAMMA * P)

        # 2.6 Calculate system total energy
        integral_1 = np.sum(M1 * phi_h)
        integral_2 = np.sum(M2 * phi_h)
        integral_3 = np.sum(_g(_grad(phi_h)))

        if iteration > 1:
            energy_old = energy
            energy = -ALPHA1 * integral_1 - ALPHA2 * integral_2 \
                     + BETA * integral_3 + GAMMA * np.sum(P)
            e_change = (energy - energy_old) / energy_old * 100.0
            print ("Total energy: {0:0.04f}, decrement: {1:0.02f} %. "
                   "Elapsed time: {2:0.01f} s.".format(energy, e_change,
                                                       time()-start_t))
            if np.abs(e_change) < PERCENT_CONVERGENCE:
                break
        else:
            energy = -ALPHA1 * integral_1 - ALPHA2 * integral_2 \
                     + BETA * integral_3 + GAMMA * np.sum(P)
            print ("Total energy: {0:0.04f}. Elapsed time: {1:0.01f} s."
                   .format(energy, time()-start_t))

    return phi


def main(args):
    """Main program code."""

    # Check image list
    imfiles = []
    for imfile in args:
        try:
            _ = nib.load(imfile).header   # low-cost load operation
            imfiles.append(imfile)
        except:
            print ("SKIPPED: {} could not be opened.".format(imfile))
            continue

    # PROCESS THE IMAGES
    for imfile in imfiles:

        # Load image
        mri = nib.load(imfile)
        hdr = mri.header
        img = mri.get_data()

        _kernel(img, 2)

        contour = acm(img)
        np.save("output/contour.npy", contour)

        plt.imshow(contour[:,:,6], cmap='gray')
        plt.show()

    else:
        print ("All tasks were successfully completed.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1:])
    else:
        print (usermanual)
        print ("\nPlease specify an image in the command-line arguments.")
        exit(0)

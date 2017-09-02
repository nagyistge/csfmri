#!/usr/bin/env python

# IMPORTS

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from sklearn.mixture import GaussianMixture
import sys
from scipy.spatial.distance import euclidean


# DESCRIPTION

usermanual = \
    """This utility tries to segment a 3D TOF (time-of-flight) MR image to 
create an MR Angiogram (MRA). The algorithm used is the GMM + MRF. It requires 
a bias-field corrected image. Please use fsl_anat to remove the receiver bias 
field from the images."""

# This module implements the algorithm originally described in:
# Yun Tian, Qingli Chen, Wei Wang, et al., “A Vessel Active Contour Model for
# Vascular Segmentation,” BioMed Research International, vol. 2014, Article ID
# 106490, 15 pages, 2014. doi:10.1155/2014/106490


# DEFINITIONS AND CODE

# Radius of neighbourhood
NB = 1
EPSILON = 0.05
RHO = 2
# Weighting factors for first- and second-order structuredness in the
# calculation of the vesselness score.
ALPHA = 17.96
BETA = 10.664
# Threshold for vesselness score
TAU = 0.05
# Coefficients for gradient descent
NU = 0.2
MU = 1
# Coefficient for the temporal evolution of phi
LAMBDA = -0.1
# Time increment
DT = 0.1


def _vessel_boundaries(vessels, nb=NB):
    comparisons = []
    for ix in np.arange(-nb, nb+1, 1):
        for iy in np.arange(-nb, nb+1, 1):
            for iz in np.arange(-nb, nb+1, 1):
                if not (ix, iy, iz) == (0, 0, 0):
                    tmp = np.roll(np.roll(np.roll(vessels, ix, axis=0),
                                          iy, axis=1), iz, axis=2)
                    comparisons.append((vessels - tmp).astype(np.bool))
    comparisons = np.stack(comparisons, axis=3).astype(np.int8)
    threshold = int(0.0 * comparisons.shape[-1])
    return vessels * np.where(np.sum(comparisons, axis=-1) > threshold, 1, 0)


def _kernel(sigma, nb=NB):
    """Gaussian kernel function representing distance score."""
    kernelvals = np.zeros((2 * nb + 1,) * 3)
    for ix in np.arange(-nb, nb+1, 1):
        for iy in np.arange(-nb, nb+1, 1):
            for iz in np.arange(-nb, nb+1, 1):
                if not (ix, iy, iz) == (0, 0, 0):
                    kernelvals[ix+nb, iy+nb, iz+nb] = \
                        np.exp(np.linalg.norm([ix, iy, iz]) ** 2 /
                               (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)
    return kernelvals


def _h(x, epsilon=EPSILON):
    """Quasi-smooth Heaviside function"""
    res = np.where(x < epsilon, 0, 1)
    res[np.where(x == epsilon)] = \
        0.5 + np.arctan(x[np.where(x == epsilon)] / epsilon) / np.pi
    return res


def _phi(vessels, boundaries, rho=RHO):
    """Calculates the level set function phi."""
    res = np.full(vessels.shape, rho)
    res[np.where(vessels)] = -res[np.where(vessels)]
    res[np.where(boundaries)] = 0
    return res.astype(np.float32)


def _calcf1f2(img, phi, kernel, epsilon=EPSILON, nb=NB):
    """Calculates the values of f1 and f2."""
    f1_numerator = []
    f2_numerator = []
    f1_denominator = []
    f2_denominator = []
    for ix in np.arange(-nb, nb + 1, 1):
        for iy in np.arange(-nb, nb + 1, 1):
            for iz in np.arange(-nb, nb + 1, 1):
                if not (ix, iy, iz) == (0, 0, 0):
                    int_y = np.roll(np.roll(np.roll(img, ix, axis=0),
                                        iy, axis=1), iz, axis=2)
                    phi_y = np.roll(np.roll(np.roll(phi, ix, axis=0),
                                        iy, axis=1), iz, axis=2)
                    f1_numerator.append(kernel[ix+nb, iy+nb, iz+nb]
                                        * _h(phi_y, epsilon) * int_y)
                    f1_denominator.append(kernel[ix+nb, iy+nb, iz+nb]
                                          * _h(phi_y, epsilon))
                    f2_numerator.append(kernel[ix+nb, iy+nb, iz+nb]
                                        * (1 - _h(phi_y, epsilon)) * int_y)
                    f2_denominator.append(kernel[ix+nb, iy+nb, iz+nb]
                                          * (1 - _h(phi_y, epsilon)))
    f1_numerator = np.sum(np.stack(f1_numerator, axis=3), axis=-1)
    f1_denominator = np.sum(np.stack(f1_denominator, axis=3), axis=-1)
    f1 = f1_numerator / f1_denominator
    f1[~np.isfinite(f1)] = 0
    f2_numerator = np.sum(np.stack(f2_numerator, axis=3), axis=-1)
    f2_denominator = np.sum(np.stack(f2_denominator, axis=3), axis=-1)
    f2 = f2_numerator / f2_denominator
    f2[~np.isfinite(f2)] = 0

    return f1, f2


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


def _vesselness(Rb, S, eigvals, alpha=ALPHA, beta=BETA):
    """Calculates vesselness score based on indicators of structuredness."""
    res = np.zeros_like(Rb)
    roi = np.where(eigvals[..., 1] <= 0)
    res[roi] = np.exp(-Rb[roi] ** 2 / (2 * alpha ** 2)) \
               * (1 - np.exp(-S[roi] ** 2 / (2 * beta ** 2)))
    return res


def _init_vvf(R, eigvects, tau=TAU):
    """Calculates vascular vector field based on vesselness index."""
    res = np.zeros(R.shape + (eigvects.shape[-2],))
    roi = np.where(R > tau)
    res[roi] = eigvects[roi][:, 0]
    return res


def _update_vvf(V, phi):
    grad_phi = np.stack(np.gradient(phi), axis=phi.ndim)
    scalarprod = np.sum(V * grad_phi, axis=-1)
    roi = np.where(scalarprod < 0)
    V[roi] = -V[roi]
    return V


def _delta(x, epsilon=EPSILON):
    """Delta function"""
    return (epsilon / (epsilon ** 2 + x ** 2)) / np.pi


def _div(field):
    """Calculates the divergence of a vector field."""
    return np.sum([np.gradient(field[..., i])[i]
                   for i in range(field.shape[-1])])


def _laplacian(field):
    """Calculates the Laplacian of an n-dimensional scalar field."""
    return np.trace(_hessian(field), axis1=field.ndim, axis2=field.ndim+1)


def _descent(img, phi, kernel, f1, f2, nb=NB, nu=NU, mu=MU):
    """Calculate the gradient descent for phi."""

    nb_integral = []
    for ix in np.arange(-nb, nb + 1, 1):
        for iy in np.arange(-nb, nb + 1, 1):
            for iz in np.arange(-nb, nb + 1, 1):
                if not (ix, iy, iz) == (0, 0, 0):
                    int_y = np.roll(np.roll(np.roll(img, ix, axis=0),
                                            iy, axis=1), iz, axis=2)
                    phi_y = np.roll(np.roll(np.roll(phi, ix, axis=0),
                                            iy, axis=1), iz, axis=2)
                    nb_integral.append(kernel[ix+nb, iy+nb, iz+nb] * _delta(phi_y) *
                                       ((int_y - f1) ** 2 - (int_y - f2) ** 2))
    nb_integral = np.sum(np.stack(nb_integral, axis=3), axis=-1)
    grad_phi = np.stack(np.gradient(phi), axis=3)
    norm_grad_phi = \
        grad_phi / np.repeat(np.linalg.norm(grad_phi, axis=3)[..., np.newaxis],
                             repeats=grad_phi.shape[-1], axis=-1,)
    norm_grad_phi[~np.isfinite(norm_grad_phi)] = 0
    term_2 = _delta(phi) * _div(norm_grad_phi)
    term_3 = _laplacian(phi) - _div(norm_grad_phi)

    grad_descent = _delta(phi) * nb_integral + nu * term_2 + mu * term_3
    grad_descent[~np.isfinite(grad_descent)] = 0

    return grad_descent


def _update_phi(phi, grad_descent, evol_speed, vvf, dt=DT, _lambda=LAMBDA):
    """Updates phi."""
    grad_phi = np.stack(np.gradient(phi), axis=phi.ndim)
    scalar_prod = np.sum(vvf * grad_phi, axis=-1)
    phi += dt * (grad_descent + _lambda * evol_speed * np.abs(scalar_prod))
    return phi


def _total_energy(img, phi, kernel, f1, f2, epsilon=EPSILON, nu=NU, mu=MU,
                  nb=NB):
    """Calculates the current total energy of the active contour."""

    nb_integral = []
    for ix in np.arange(-nb, nb + 1, 1):
        for iy in np.arange(-nb, nb + 1, 1):
            for iz in np.arange(-nb, nb + 1, 1):
                if not (ix, iy, iz) == (0, 0, 0):
                    int_y = np.roll(np.roll(np.roll(img, ix, axis=0),
                                            iy, axis=1), iz, axis=2)
                    phi_y = np.roll(np.roll(np.roll(phi, ix, axis=0),
                                            iy, axis=1), iz, axis=2)
                    nb_integral.append(
                        kernel[ix+nb, iy+nb, iz+nb] *
                        ((_h(phi_y) * (int_y - f1) ** 2)
                         + ((1 - _h(phi_y)) * (int_y - f2) ** 2))
                    )
    nb_integral = np.sum(np.stack(nb_integral, axis=3), axis=-1)
    grad_phi = np.stack(np.gradient(phi), axis=phi.ndim)
    integral_2 = np.sum(_delta(phi, epsilon)
                        * np.linalg.norm(grad_phi, axis=-1))
    integral_3 = np.sum(0.5 * (np.linalg.norm(grad_phi, axis=-1) - 1) ** 2)

    return np.sum(_delta(phi) * nb_integral) + nu * integral_2 + mu * integral_3


def acm(tofimg):
    """
    :param ndarray tofimg: 3D bias-corrected TOF (time-of-flight) image.
    """

    # Initialise vessel locations and their approximate boundaries
    vessels = np.zeros_like(tofimg)
    vessels[tofimg > np.percentile(tofimg[np.nonzero(tofimg)], 99)] = 1
    boundaries = _vessel_boundaries(vessels, nb=NB)

    # Initialising the level set function phi
    phi = _phi(vessels, boundaries, rho=RHO)

    # Calculate kernel
    kernel = _kernel(sigma=np.sum(tofimg.shape)/16.0, nb=NB)

    # Calculate vesselness index (R) from the Hessian of the the image
    eigvals, eigvects = np.linalg.eig(_hessian(tofimg))
    Rb = np.abs(eigvals[..., 0].astype(np.float64)) \
                       / np.abs(eigvals[..., 1].astype(np.float64))
    Rb[~np.isfinite(Rb)] = 0
    S = np.linalg.norm(eigvals[..., 0:2].astype(np.float64), axis=-1)
    R = _vesselness(Rb, S, eigvals, alpha=ALPHA, beta=BETA)

    # Define contour evolution speed function
    evol_speed = 0.5 + np.arctan(R / EPSILON - 1) / np.pi

    # Initialise the vascular vector field
    V = _init_vvf(R, eigvects, tau=TAU)

    i = 0
    E = None

    while True:

        # Update status
        i += 1
        print ("Performing iteration No. {}...".format(i))

        # Update f1 and f2
        f1, f2 = _calcf1f2(tofimg, phi, kernel, epsilon=EPSILON, nb=NB)

        # Update the vascular vector field
        V = _update_vvf(V, phi)

        # Calculate the gradient decent
        gd = _descent(tofimg, phi, kernel, f1, f2, nb=NB, nu=NU, mu=MU)

        # Update phi
        phi = _update_phi(phi, gd, evol_speed, V, dt=DT, _lambda=LAMBDA)

        # Calculate energy and convergence

        if E is not None:
            E_previous = E
            E = _total_energy(tofimg, phi, kernel, f1, f2, epsilon=EPSILON,
                              nu=NU, mu=MU, nb=NB)
            dec = (E - E_previous) / E_previous * 100.0
        else:
            E = _total_energy(tofimg, phi, kernel, f1, f2, epsilon=EPSILON,
                              nu=NU, mu=MU, nb=NB)
            dec = np.nan

        print ("Total energy: {0:0.04f}, decrement: {1:0.02f} %."
               .format(E, dec))

        if np.abs(dec) < 1:
            break

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

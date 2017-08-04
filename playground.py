#!/Volumes/INH_1TB/CSFMRI/venv/venv_csfmri/bin/python

# DESCRIPTION

"""This Python module is intended for testing code and visualising data for
development purposes."""

# IMPORTS

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks_cwt
import multiprocessing
from time import time
from joblib import Parallel, delayed
from scipy.io import loadmat
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.feature_selection import f_regression
from statsmodels.genmod import generalized_linear_model
import statsmodels.api as sm
import statsmodels

# DEFINITIONS AND CODE

def show_biodata(sourcefile):
    biodata = np.loadtxt(sourcefile)
    print biodata.shape
    """
    fig1, ax1 = plt.subplots(2, 2)
    ax1[0, 0].plot(biodata[:, 0])    # respiratory
    ax1[0, 1].plot(biodata[:, 1])    # cardiac
    ax1[1, 0].plot(biodata[:, 2])    # triggger
    ax1[1, 1].plot(biodata[:, 3])    # sats
    """

    # Demean signal
    cardsignal = biodata[:, 2] - np.mean(biodata[:, 2])

    # Show signal
    """
    plt.plot(biodata[:, 1])
    plt.show()
    exit()
    """

    # Discrete Fourier transform (sampling rate: 1kHz)
    voxelsignals = np.repeat(cardsignal[np.newaxis, :], 100, axis=0)
    print voxelsignals.shape
    voxelcardspectra = np.abs(np.fft.fft(voxelsignals))
    freq = np.fft.fftfreq(len(cardsignal[0, :]), d=0.001)
    for sp in range(0,100):
        plt.plot(freq[:len(freq) / 2], voxelcardspectra[sp, :len(freq) / 2])
    exit()

    """
    cardspectrum = np.abs(np.fft.fft(cardsignal))
    freq = np.fft.fftfreq(len(cardsignal), d=0.001)
    
    plt.figure()
    plt.plot(freq[:len(freq)/2], cardspectrum[:len(freq)/2])
    plt.show()

    max_index = np.argmax(cardspectrum[:len(freq)/2])
    max_freq = freq[max_index]
    period = 1.0/max_freq   # in seconds
    """

    # Find peaks with this period
    """
    peakloc = find_peaks_cwt(vector=cardsignal, widths=1000*np.arange(0.5*period, 1.5*period))
    plt.figure()
    peaks = np.zeros_like(cardsignal)
    peaks[peakloc] = 1
    plt.scatter(np.arange(len(cardsignal)), peaks)
    plt.show()
    """
    fig2, (ax2, ax3) = plt.subplots(2, 1)
    ax2.plot(biodata[:, 0])    # respiratory
    ax3.plot(cardsignal)    # cardiac

    plt.show()


def quickfft(sourcefile):

    biodata = np.loadtxt(sourcefile)
    print "File loaded."
    cardsignal = biodata[:, 2]
    cardsignal = cardsignal - np.mean(cardsignal)
    print cardsignal[:10000:20].shape
    voxelcardsignal = np.repeat(cardsignal[np.newaxis, :10000:20], 285120, axis=0)
    singlecardsignal = cardsignal[np.newaxis, :10000:20]
    print "Large array ready."
    """
    voxelcardsignal = np.zeros((90, 88, 36, 500))
    start_t = time()
    for i in np.arange(0, 90):
        for j in np.arange(0, 88):
            for k in np.arange(0, 36):
                voxelcardsignal[i, j, k, :] = np.fft.fft(singlecardsignal)
    print time() - start_t"""
    start_t = time()
    voxelcardspectra = np.fft.rfft(voxelcardsignal)
    end_t = time()
    print end_t - start_t
    print np.array(voxelcardspectra).shape
    """
    start_ts = time()
    voxelcardspectra = np.fft.fft([vcs for vcs in voxelcardsignal])
    print time() - start_ts
    print np.array(voxelcardspectra).shape
    print "Creating parallel processess..."
    parpool = multiprocessing.Pool(2)
    print "Calculating FFTs..."
    start_ts = time()
    voxelcardspectra = parpool.map(np.fft.fft, [vcs for vcs in voxelcardsignal])
    print time() - start_ts
    print np.array(voxelcardspectra).shape
    """

def try_GLMstat(matfile):
    vars = loadmat(matfile)
    f = vars['fTRRefined']
    y = vars['AVoxelSpectra']
    """
    plt.figure()
    plt.plot(f.ravel(), y[25, 47, 6, :])
    plt.show()
    """
    print "y:", y.shape
    X = vars['RefinedSpec']
    """
    plt.figure()
    plt.plot(f.ravel(), X[:, 0])
    plt.plot(f.ravel(), X[:, 1])
    plt.plot(f.ravel(), X[:, 2])
    plt.show()
    """
    print "X:", X.shape
    b = vars['SBetasRefined']
    print "b:", b.shape
    """
    plt.figure()
    plt.bar(range(3), b[25, 47, 6, :])
    plt.show()
    """
    b = b.reshape((-1, b.shape[-1])).T
    y = y.reshape((-1, y.shape[-1])).T
    print np.min(y), np.max(y)
    #print y[70:80, 50:60]
    #res = LinearRegression().fit(X, y)
    res = BayesianRidge().fit(X, y)
    print np.min(res.coef_), np.max(res.coef_)
    #np.save('output/mybetas.npy', res.coef_)
    myres = res.coef_.T
    print "Differences:"
    diff = myres - b
    print np.min(diff), np.max(diff)


from sklearn import linear_model
from scipy import stats
import numpy as np


#https://stackoverflow.com/questions/27928275/find-p-value-significance-in-scikit-learn-linearregression
class LinearRegression(linear_model.LinearRegression):
    """
    LinearRegression class after sklearn's, but calculate t-statistics
    and p-values for model coefficients (betas).
    Additional attributes available after .fit()
    are `t` and `p` which are of the shape (y.shape[1], X.shape[1])
    which is (n_features, n_coefs)
    This class sets the intercept to 0 by default, since usually we include it
    in X.
    """

    def __init__(self, *args, **kwargs):
        if not "fit_intercept" in kwargs:
            kwargs['fit_intercept'] = False
        super(LinearRegression, self)\
                .__init__(*args, **kwargs)

    def fit(self, X, y, n_jobs=1):
        self = super(LinearRegression, self).fit(X, y, n_jobs)

        sse = np.sum((self.predict(X) - y) ** 2, axis=0) / float(X.shape[0] - X.shape[1])
        se = np.array([
            np.sqrt(np.diagonal(sse[i] * np.linalg.inv(np.dot(X.T, X))))
                                                    for i in range(sse.shape[0])
                    ])

        self.t = self.coef_ / se
        self.p = 2 * (1 - stats.t.cdf(np.abs(self.t), y.shape[0] - X.shape[1]))
        return self

vars = loadmat("/Users/inhuszar/csfmri/analysis_matlab/F3T_2013_40_363/"
         "variables/RefinedData.mat")
X = vars['RefinedSpec']
#sm_probit_Link = statsmodels.genmod.families.links.identity


def glmfunc(voxel):
    #glm = sm.GLM(voxel, X,family=sm.families.Gaussian(link=sm_probit_Link))\
    #      .fit(use_t=True)
    #glm = generalized_linear_model.GLM(voxel, X).fit(use_t=True)
    return glm.coef_, glm.p


def GLMpystat(matfile):
    #vars = loadmat(matfile)
    f = vars['fTRRefined']
    y = vars['AVoxelSpectra']
    print "y:", y.shape
    X = vars['RefinedSpec']
    print "X:", X.shape
    b = vars['SBetasRefined']
    print "b:", b.shape
    y = y.reshape((-1, y.shape[-1]))
    print ("Calculating ROI...")
    y_roi = y[np.sum(y, axis=1) > 0, :]

    print ("Initializing parallel processes...")
    #parpool = multiprocessing.Pool(4)
    print ("Starting the job...")
    start_t = time()
    #X = np.hstack((np.ones_like(X[:, 1]).reshape(-1, 1), X[:, 1:]))
    glm = LinearRegression(n_jobs=-1).fit(X, y_roi.T)
    end_t = time()
    print "0th param significant:", glm.p[glm.p[:, 0] < 0.05].shape[0]
    print "1st param significant:", glm.p[glm.p[:, 1] < 0.05].shape[0]
    print "2nd param significant:", glm.p[glm.p[:, 2] < 0.05].shape[0]
    print "Both params significant:", glm.p[np.logical_and(glm.p[:, 1] < 0.05, glm.p[:, 2] < 0.05)].shape[0]
    #glm_res = parpool.map(glmfunc, [voxel for voxel in y_roi])
    print ("Elapsed time: {}".format(end_t-start_t))
    print "Number of GLM fits:", glm.coef_.shape[0]
    print "R^2:", 1 - np.sum((y_roi.T - np.dot(X, glm.coef_.T)) ** 2) / np.sum(
        (y_roi.T - np.mean(y_roi.T, axis=0)) ** 2)

    # Compare with Matlab
    b = b.reshape((-1, b.shape[-1]))
    print "0th param significant:", b[b[:, 0] > 0].shape[0]
    print "1st param significant:", b[b[:, 1] > 0].shape[0]
    print "2nd param significant:", b[b[:, 2] > 0].shape[0]
    print "Both params significant:", \
    b[np.logical_and(b[:, 1] > 0, b[:, 2] > 0)].shape[0]
    print "R^2:", 1 - np.sum((y.T - np.dot(X,b.T)) ** 2) / np.sum((y.T - np.mean(y.T, axis=0)) ** 2)
    """
    glm = [generalized_linear_model.GLM(voxel, X).fit()
           for voxel in y_roi]
    """


def reproduceGLM():
    """Trying to reproduce the results of the first GLM code in Second_GLM.m."""
    # Constants
    TR = 0.308
    Sfreq = 1000
    fmin = 0.2

    # Import physiological data
    biodata = np.loadtxt("/Users/inhuszar/csfmri/data/F3T_2013_40_363/"
                         "2017_07_18_singleEcho.txt")

    # Sub-sample cardiac and respiratory signal at 1/TR.


if __name__ == "__main__":
    """
    quickfft(
        "/Users/inhuszar/csfmri/data/F3T_2013_40_363/2017_07_18_singleEcho.txt")
    try_GLMstat("/Users/inhuszar/csfmri/analysis_matlab/F3T_2013_40_363/"
             "variables/RefinedData.mat")
    """
    GLMpystat("/Users/inhuszar/csfmri/analysis_matlab/F3T_2013_40_363/"
             "variables/RefinedData.mat")
    #reproduceGLM()
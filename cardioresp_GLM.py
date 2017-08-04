#!/Volumes/INH_1TB/CSFMRI/venv/venv_csfmri/bin/python

################################################################################
# This script was created part of the CSFMRI (cardio-synchronous fMRI) project #
# at the University of Oxford, Centre for Functional Magnetic Resonance        #
# Imaging of the Brain (FMRIB).                                                #
#                                                                              #
# Principal investigator: Professor Peter Jezzard                              #
#                         (peter.jezzard@univ.ox.ac.uk)                        #
#                                                                              #
# The original data analysis pipeline was created in Matlab                    #
# by Olivia Viessmann (olivia.viessmann@trinity.ox.ac.uk).                     #
#                                                                              #
# Author: Istvan N. Huszar, M.D. (istvan.huszar@dtc.ox.ac.uk)                  #
# Date: 2017-Aug-02                                                            #
################################################################################

# DESCRIPTION
# FIXME: Make this module also executable. Update the user information as well.

usermanual = \
    """ OBSOLETE !!
    This Python module contains a sophisticated GLM fitting sub-routine that 
    can be used to decompose the BOLD signal into a linear combination of the 
    cardiac and respiratory signals. If either (or both) is not available, the 
    algorithm tries to infer them from the compound signal. The sub-routine can 
    also be run from the command line.
    
    Usage:
        {CSFMRI}/cardioresp_GLM.py -r <func_res> -f <freq_range> 
        -l <signal_resp> -h <signal_card> -p <p_value> [-c <convergence>]
    
    INPUT:
        -r <func_res>        Functional image residuals (from FEAT) (4D array)
        -f <freq_range>      Frequency range (1D array) (trimmed adequately)
        -l <signal_resp>     Respiratory signal (1D array)
        -h <signal_card>     Cardiac signal (1D array)
        [-p <p_value>]       Significance level. Default: 0.05
        [-c <convergence>]   Iterations will stop when the summed differences of 
                           the previous and current spectra falls below this 
                           level. Default: 0.1
    
        Note: all input signals must have the same temporal resolution. p-value 
        and convergence are optional parameters. If not specified, defaults are 
        used.
    
    OUTPUTS:
        betas_refined   Final parameter estimated (0: respiratory, 1: cardiac)
        betas_init      Initial parameter estimates (0: respiratory, 1: cardiac)
        spectra_refined 0: Voxel-wise BOLD spectra for the whole FOV
                        1: Refined respiratory power spectrum
                        2: Refined cardiac power spectrum
                        3: Refined baseline power spectrum
    """

# IMPORTS

from cl_interface import *
from csfmri_exceptions import *
import numpy as np
from sklearn import linear_model
from scipy import stats

# DEFINITIONS AND CODE

# Command-line arguments
# OBSOLETE !!!
CLFLAGS = {'func_res': '-r',        # r(residuals)
           'freq_range': '-f',      # f(requencies)
           'signal_resp': '-l',     # l(ung)
           'signal_card': '-h',     # h(eart)
           'p_val': '-p',           # p( value)
           'convergence': '-c'}     # c(onvergence)


class InputDescriptorObj:
    def __init__(self):
        pass


# Source: https://stackoverflow.com/questions/27928275/find-p-value-significance
# -in-scikit-learn-linearregression
class LinearRegression(linear_model.LinearRegression):
    """
    LinearRegression class after sklearn's, but calculate t-statistics and
    p-values for model coefficients (betas).
    Additional attributes available after .fit() are `t` and `p` which are of
    the shape (y.shape[1], X.shape[1]) which is (n_features, n_coefs).
    This class sets the intercept to 0 by default, since usually we include it
    in X.
    """

    def __init__(self, *args, **kwargs):
        if "fit_intercept" not in kwargs:
            kwargs['fit_intercept'] = False
        super(LinearRegression, self)\
            .__init__(*args, **kwargs)

    def fit(self, X, y, n_jobs=1):
        self = super(LinearRegression, self).fit(X, y, n_jobs)

        sse = np.sum((self.predict(X) - y) ** 2, axis=0) / \
            float(X.shape[0] - X.shape[1])
        se = np.array([np.sqrt(
             np.diagonal(sse[i] * np.linalg.inv(np.dot(X.T, X))))
             for i in range(sse.shape[0])])

        self.t = self.coef_ / se
        self.p = 2 * (1 - stats.t.cdf(np.abs(self.t), y.shape[0] - X.shape[1]))
        return self


class GLMObject:
    """This is an object that performs the iterative GLM. The output is stored
    in a GLMFitObject. It is capable of handling n EVs (explanatory variables).
    """

    # Constructor
    def __init__(self, EV_, signal_, sfreq_=1, fmin_=0, pval_=0.05,
                 clevel_=0.1):
        self.signal = signal_
        self.EV = EV_
        self.sfreq = sfreq_
        self.fmin = fmin_
        self.pval = pval_
        self.clevel = clevel_

    # Properties
    @property
    def signal(self):
        return self.__signal

    @signal.setter
    def signal(self, signal_):
        try:
            self.__signal = np.array(signal_)
            assert np.issubdtype(self.__signal.dtype, np.number)
        except:
            raise TypeMismatchException("GLMObject: invalid input for signal.")

    @property
    def EV(self):
        return self.__EV

    @EV.setter
    def EV(self, EV_):
        try:
            self.__EV = np.array(EV_)
            assert np.issubdtype(self.__signal.dtype, np.number)
        except:
            raise TypeMismatchException("GLMObject: the input for explanatory "
                                        "variables must be numerical.")
        try:
            assert self.__EV.shape[0] == self.__signal.shape[0]
        except:
            raise CountsMismatchException("GLMObject: Explanatory variables "
                                          "must have the same size as the "
                                          "signal.")

    @property
    def sfreq(self):
        return self.__sfreq

    @sfreq.setter
    def sfreq(self, sfreq_):
        try:
            self.__sfreq = float(sfreq_)
        except:
            raise TypeMismatchException("GLMObject: Sampling frequency must be "
                                        "real-valued.")

    @property
    def fmin(self):
        return self.__fmin

    @fmin.setter
    def fmin(self, fmin_):
        try:
            self.__fmin = float(fmin_)
            assert self.__fmin >= 0
        except:
            raise ValueError("GLMObject: Lower cut-off frequency must be "
                             "real-valued and non-negative.")

    @property
    def pval(self):
        return self.__pval

    @pval.setter
    def pval(self, pval_):
        try:
            self.__pval = float(pval_)
            assert (self.__pval <= 0) and (self.__pval <= 1)
        except:
            raise ValueError("GLMObject: Significance level (p value) must be "
                             "in the range [0,1].")

    @property
    def clevel(self):
        return self.__clevel

    @clevel.setter
    def clevel(self, clevel_):
        try:
            self.__clevel = float(clevel_)
            assert self.__clevel >= 0
        except:
            raise ValueError("GLMObject: Convergence level must be real-valued "
                             "and non-negative.")

    # Methods
    @staticmethod
    def _opt_fourier(input_, axis_=-1, abs_=True, fmin=0.0, sfreq=1.0):
        """This method runs optimized Discrete Fourier Transform based on the
        input data."""
        data = np.asarray(input_)
        if np.issubdtype(data.dtype, np.complex_):
            fft_result = np.fft.fft(data, axis=axis_)
            fft_freq = np.fft.fftfreq(data.shape[axis_], 1.0/sfreq)
        else:
            fft_result = np.fft.rfft(data, axis=axis_)
            fft_freq = np.fft.rfftfreq(data.shape[axis_], 1.0/sfreq)
        if fmin != 0:
            fft_freq = fft_freq[np.abs(fft_freq) > fmin]
            fft_result = fft_result[np.abs(fft_freq) > fmin]
        if abs_:
            return fft_freq, np.abs(fft_result)
        else:
            return fft_freq, fft_result

    def fit(self, n_jobs_=-1, total_n_EVs=None, iterations=True, normalize=True,
            verbose=True):
        """Fits linear model according to the settings.
        Parameters
        ----------
            :param n_jobs_: Number of CPU cores used for fitting. (-1: all)
            :param total_n_EVs: Total number of desired explanatory variables
                                (EVs).
            :param iterations: Iterations will be performed if True.
            :param normalize: Normalize the power spectra of the voxels and the
                              EVs.
            :param verbose: Display infromation about progress.
        """
        # TODO: The object's fitting functionality could be improved by adding
        #       the option for using alternative methods.
        # TODO: Use appropriate documentation markup.

        # Check the format of the signal: flatten to 2D if necessary
        signal_shape = self.__signal.shape
        if len(signal_shape) > 2:
            self.__signal = self.__signal.reshape((-1, signal_shape[-1]))

        # Calculate Fourier spectra of each signal
        # Note: signal matrix: voxels (rows) x timepoints (columns),
        #       design matrix (EVs): timepoints (rows) x EVs (columns)
        # The second computation of the FFT frequencies vector is discarded,
        # being identical to the first.
        fft_frequencies, fft_signals = GLMObject._opt_fourier(
            self.__signal, axis_=-1, abs_=True, fmin=self.__fmin,
            sfreq=self.__sfreq)
        _, fft_EVs = GLMObject._opt_fourier(
            self.__EV, axis_=0, abs_=True, fmin=self.__fmin, sfreq=self.__sfreq)

        # Normalize the signal spectrum and all EV spectra (fmin:Nyquist)
        if normalize:
            fft_signals = fft_signals / np.sum(fft_signals, axis=-1)
            fft_EVs = fft_EVs / np.sum(fft_EVs, axis=0)

        # Discard any false signals (all zero over time, like background)
        true_signal_mask = np.any(fft_signals, axis=1)
        fft_signals_nonzero = fft_signals[true_signal_mask, :]
        if verbose:
            print ("{} non-zero voxels will be used out of {}."
                   .format(fft_signals.shape[0], fft_signals_nonzero.shape[0]))
        # Assume that there is no such EVs...
        # true_EV_mask = np.any(fft_EVs, axis=0)
        # fft_EVs_nonzero = fft_EVs[:, true_EV_mask]

        # Determine whether any of the EVs is missing from the initial
        # specification. The missing EVs will be created sequentially from the
        # baseline fit (column of constants) after each iteration.
        if total_n_EVs:
            try:
                total_n_EVs = int(total_n_EVs)
                n_missing_EVs = total_n_EVs - fft_EVs.shape[-1]
                assert n_missing_EVs >= 0
            except:
                raise ValueError(
                    "GLMObject: The total number of explanatory variables has "
                    "to be an integer that is greater than or equal to the "
                    "number of explanatory variable initially provided.")
        else:
            n_missing_EVs = 0

        # Perform iterative fitting until the desired level of convergence is
        # reached. Initialise convergence level and iteration counter then start
        # iterations.
        convergence = self.__clevel + 1
        current_iteration = 0
        while convergence >= self.__clevel:
            # Step iteration counter.
            current_iteration += 1
            if verbose:
                print ("Started {}. iteration.".format(current_iteration))

            # Perform initial GLM fitting using all CPU cores
            # Note that fft_signals_nonzero has to be transposed into the form
            # (n_samples x n_targets).
            if verbose:
                print ("Fitting initial GLM...")
            # Add a single new column of ones from the left to the EV matrix
            # (if necessary):
            if n_missing_EVs:
                fft_EVs_initial = np.hstack((np.ones((fft_EVs.shape[0], 1)),
                                             fft_EVs))
                if normalize:
                    fft_EVs_initial = fft_EVs_initial / \
                                      np.sum(fft_EVs_initial, axis=0)
                n_missing_EVs -= 1
            else:
                fft_EVs_initial = np.copy(fft_EVs)

            # Perform the initial GLM.
            initial_glm = LinearRegression(n_jobs=n_jobs_)\
                          .fit(fft_EVs_initial, fft_signals_nonzero.T)
            if verbose:
                print ("Initial GLM fit finished.")

            # Set any negative and NaN coefficients to zero.
            negative_coef_mask = initial_glm.coef_ < 0
            if np.any(negative_coef_mask):
                print ("WARNING: Negative coefficients after initial GLM were "
                       "set to zero.")
                initial_glm.coef_[negative_coef_mask] = 0
            nan_coef_mask = np.isnan(initial_glm.coef_)
            if np.any(nan_coef_mask):
                print ("WARNING: NaN coefficients after initial GLM were "
                       "set to zero.")
                initial_glm.coef_[nan_coef_mask] = 0

            # Only keep coefficients that are significantly different from 0.
            initial_glm.coef_[initial_glm.p >= self.__pval] = 0

            # Refine explanatory variables by fitting the significant
            # coefficients from the initial GLM to a given frequency component
            # in all voxels at the same time. Iterating this for all frequencies
            # leads to new explanatory spectra, which are thought to be
            # "refined" versions of the previous set of explanatory variables.
            # This GLM is almost identical to the initial one but the fft_signal
            # matrix is not transposed, hence it is used column-wise and the
            # coefficient matrix is used instead of fft_EVs matrix. Also, this
            # time no column of ones has to be added to the design matrix.
            if verbose:
                print ("Fitting spatial GLM...")
            spatial_glm = LinearRegression(n_jobs=n_jobs_)\
                .fit(initial_glm.coef_, fft_signals_nonzero)
            if verbose:
                print ("Spatial GLM fit finished.")

            # Set any negative and NaN coefficients to zero.
            negative_coef_mask = spatial_glm.coef_ < 0
            if np.any(negative_coef_mask):
                print ("WARNING: Negative coefficients after spatial GLM were "
                       "set to zero.")
                spatial_glm.coef_[negative_coef_mask] = 0
            nan_coef_mask = np.isnan(spatial_glm.coef_)
            if np.any(nan_coef_mask):
                print ("WARNING: NaN coefficients after spatial GLM were "
                       "set to zero.")
                spatial_glm.coef_[nan_coef_mask] = 0

            # Only keep coefficients that are significantly different from 0.
            spatial_glm.coef_[spatial_glm.p >= self.__pval] = 0

            # Final step in each iteration: re-run the initial GLM fit using the
            # refined explanatory variables (spectra).
            if verbose:
                print ("Re-fitting GLM using the refined explanatory "
                       "variables.")
            # Rename container variable to avoid confusion. The coefficients of
            # the spatial GLM build up the refined spectra. The matrix has to be
            # transposed before being used as the design matrix of the refined
            # GLM. No column of ones has to be added.
            fft_EVs_refined = spatial_glm.coef_.T
            refined_glm = LinearRegression(n_jobs=n_jobs_)\
                .fit(fft_EVs_refined, fft_signals_nonzero.T)
            if verbose:
                print ("Refined GLM fit finished.")

            # Set any negative and NaN coefficients to zero.
            negative_coef_mask = refined_glm.coef_ < 0
            if np.any(negative_coef_mask):
                print ("WARNING: Negative coefficients after refined GLM were "
                       "set to zero.")
                refined_glm.coef_[negative_coef_mask] = 0
            nan_coef_mask = np.isnan(refined_glm.coef_)
            if np.any(nan_coef_mask):
                print ("WARNING: NaN coefficients after refined GLM were set "
                       "to zero.")
                refined_glm.coef_[nan_coef_mask] = 0

            # Only keep coefficients that are significantly different from 0.
            refined_glm.coef_[initial_glm.p >= self.__pval] = 0

            # Calculate convergence after the current iteration
            convergence = np.max(np.sum(
                          np.abs(fft_EVs_initial - fft_EVs_refined), axis=0))
            if verbose:
                print ("Iteration {} done. Convergence = {}"
                       .format(current_iteration, convergence))

            # Update explanatory variables with the normalised refined spectra
            fft_EVs = fft_EVs_refined / np.sum(fft_EVs_refined, axis=0)

            # Exit the loop if the iterations option has been set to False.
            if not iterations:
                break
        if verbose:
            print ("Iterative GLM fitting is complete. {} iterations were "
                   "performed. The final convergence value was {}."
                   .format(current_iteration, convergence))

        # Create output: reformat initial and refined regression coefficients
        # into the (probably 4D) shape of the original signal. Note that
        # non-significant values were set to 0. Also take into account that
        # background was removed.
        # Create 2D template
        coef_initial = np.zeros((fft_signals.shape[:-1] +
                                 (fft_EVs_initial.shape[-1],)))
        coef_refined = np.zeros((fft_signals.shape[:-1] +
                                 (fft_EVs_refined.shape[-1],)))
        # Fill coefficients for non-zero voxels
        coef_initial[true_signal_mask, :] = initial_glm.coef_
        coef_refined[true_signal_mask, :] = refined_glm.coef_
        # Reformat into original shape
        coef_initial = coef_initial.reshape((signal_shape[:-1] +
                                             (fft_EVs_initial.shape[-1],)))
        coef_refined = coef_refined.reshape((signal_shape[:-1] +
                                             (fft_EVs.shape[-1],)))

        # Return output
        # Beware that both sets of coefficients come from the last iteration.
        # This was adapted from the Matlab implementation.
        # TODO: Intentional to return the "initial" coeffs from the last iter?
        return fft_signals, fft_EVs, coef_initial, coef_refined


def parse_arguments():
    """Subroutine that understands command-line arguments and passes the
    information to the main program."""


def summarize(InputObj):
    """This sub-routine prints the summary on the screen."""


def GLM():
    """This sub-routine performs the actual iterative GLM fitting of n
    explanatory variables."""


def main():
    """Main program code."""
    # Parse arguments
    InputListObj = parse_arguments()

    # Summarize task
    summarize(InputListObj)

    # Do the job


# Main program execution starts here
if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        print usermanual

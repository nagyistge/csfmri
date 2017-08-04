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
# Date: 2017-Jul-30                                                            #
################################################################################

# DESCRIPTION

usermanual = \
    """
    ============================================================================
    This Python module provides a both callable and directly executable
    sub-routine that creates the field map for EPI (echo-planar imaging)
    distortion correction.
    ============================================================================
    
    INPUTS:
        Magnitude image (nii.gz)
        Phase difference image (nii.gz)
        Echo difference (TE2 - TE1) (in milliseconds)
        Fractional intensity (FSL BET parameter, typically 0.5-0.55)
    
    OUTPUT:
        Field map image (nii.gz)
    
    Usage:
        ./field_map_BIDS.py -m <magn> -p <phasediff> -e <deltaTE> -f <fractInt>
                            [-out <>]
    
    """

# IMPORTS

from cl_interface import *
from fsl_interface import *
from csfmri_utils import *
import os
import subprocess


# DEFINITIONS AND CODE

CLFLAGS = {'magnitude': '-m',
           'phasediff': '-p',
           'echodiff': '-e',
           'fractint': '-f',
           'out': '-out',
           'interactive': '-i'}

# Default output file paths and names
# (directory and extension will be added based on the source file)
# (do not delete any of these, just set to None when needed)
FM_BASE_NAME = "FM"
FM_PATH = "../fmap"     # relative to the source directory
FM_TAG = None


class InputDescriptorObj:
    def __init__(self, MagnitudePathList, PhaseDiffPathList, EchoDiffList,
                 FractIntList, interactivity, FieldMapPathList=None):
        self.magnitude = MagnitudePathList
        self.phasediff = PhaseDiffPathList
        self.fieldmap = FieldMapPathList
        self.echodiff = EchoDiffList
        self.fractint = FractIntList
        self.interactive = interactivity

        # Check integrity of input
        inputcount = {len(self.magnitude), len(self.phasediff),
                      len(self.echodiff), len(self.fractint)}
        if self.fieldmap:
            inputcount.add(len(self.fieldmap))
        assert len(inputcount) == 1, "Integrity of the lists of input " \
                                     "(and output)"
        self.count = inputcount.pop()


def parse_arguments():
    """This sub-routine understands the command-line arguments and passes the
    information to the main code."""
    # Magnitude image
    MagnitudePathList = subarg(CLFLAGS['magnitude'])
    if not MagnitudePathList:
        print ("No magnitude image was specified.")
        exit(1)
    # Phase difference image
    PhaseDiffPathList = subarg(CLFLAGS['phasediff'])
    if not PhaseDiffPathList:
        print ("No phase difference image was specified.")
        exit(1)

    # Parse output specification
    if argexist(CLFLAGS['out'], True):
        FieldMapPathList = subarg(CLFLAGS['out'])
    else:
        print ("WARNING: Missing output specification for field map(s). The "
               "location(s) and file name(s) will be guessed automatically.")
        try:
            FieldMapPathList = guess_output_from_input(
                inputfiles=MagnitudePathList, basename=FM_BASE_NAME,
                tag=FM_TAG, subdir=FM_PATH)
        # FIXME: Placeholder for proper exception handling
        except:
            raise

    # Interactivity
    interactivity = argexist(CLFLAGS['interactive'])

    # Validate matched input (and output) images
    MagnitudePathList, PhaseDiffPathList, FieldMapPathList = \
        matched_input_validation(inputs=[MagnitudePathList, PhaseDiffPathList,
                                         FieldMapPathList],
                                 supported_formats=FORMATS.values(),
                                 interactive=interactivity)

    number_of_inputs = len(MagnitudePathList)

    # Echo difference (list)
    if argexist(CLFLAGS['echodiff'], True):
        try:
            EchoDiffList = [float(val) for val in subarg(CLFLAGS['echodiff'])]
        except:
            print ("ERROR: Encountered invalid specification for echo time "
                   "difference.")
            exit(1)
        finally:
            try:
                assert len(EchoDiffList) == number_of_inputs, \
                    "The number of echo time difference values must match the" \
                    " number of images."
            except:
                # User convenience: use the same values for all input images
                if len(EchoDiffList) == 1:
                    echodiff = EchoDiffList * number_of_inputs
                else:
                    raise

    else:
        print ("ERROR: Missing specification for echo time difference.")
        exit(1)

    # Fractional intensity (list)
    if argexist(CLFLAGS['fractint'], True):
        try:
            FractIntList = [float(val) for val in subarg(CLFLAGS['fractint'])]
            assert all([(val >= 0) and (val <= 1) for val in FractIntList])
        except:
            print ("ERROR: Encountered invalid specification for fractional "
                   "intensity.")
            exit(1)
        finally:
            try:
                assert len(FractIntList) == number_of_inputs, \
                    "The number of fractional intensity values must match the" \
                    " number of images."
            except:
                # User convenience: use the same values for all input images
                if len(FractIntList) == 1:
                    fractint = FractIntList * number_of_inputs
                else:
                    raise
    else:
        print ("ERROR: Missing specification for fractional intensity.")
        exit(1)

    return InputDescriptorObj(MagnitudePathList, PhaseDiffPathList,
                              EchoDiffList, FractIntList, interactivity,
                              FieldMapPathList)


def summarize(InputObj):
    """This sub-routine prints the summary on the screen."""
    msg = ("The following operations ({}) will be performed:"
           .format(InputObj.count))
    print msg
    print "".join(['='] * len(msg))
    for input_pack in zip(InputObj.magnitude, InputObj.phasediff,
                          InputObj.echodiff, InputObj.fractint,
                          InputObj.fieldmap):
        print ("Echo difference: {}, Fractional intensity: {}"
               .format(input_pack[2], input_pack[3]))
        print "\n".join(input_pack[:2])
        print "->"
        print input_pack[-1]
        print "\n"
    print "Interactivity: {}".format(str(InputObj.interactive))


def main():
    """Main program code. When the module is run from the command line, this
    handles the command-line arguments and communicates with the user."""

    # Parse arguments
    InputListObj = parse_arguments()

    # Summarize task
    summarize(InputListObj)
    if InputListObj.interactive:
        if not confirmed_to_proceed("Would you like to proceed? (y/n): "):
            exit(1)

    # Create output
    # FIXME: This block should allow for parallel executions.
    fsldir = get_fsldir()
    for input_index in range(InputListObj.count):
        # Run brain extraction (FSL: bet)
        betid = subprocess.Popen([os.path.join(fsldir, "/bin/bet"),
                         InputListObj.magnitude[input_index],
                         InputListObj.fieldmap[input_index] +
                         "_magnitude1.nii.gz",
                         '-f', InputListObj.fractint[input_index]], shell=True)

        # Prepare the field map (FSL: fsl_prepare_fieldmap)
        subprocess.call([os.path.join(fsldir, "/bin/fsl_prepare_fieldmap"),
                         'SIEMENS', InputListObj.phasediff[input_index],
                         InputListObj.fieldmap[input_index] +
                         "_magnitude1.nii.gz",
                         InputListObj.fieldmap[input_index] +
                         "_phasediff.nii.gz",
                         InputListObj.echodiff[input_index]], shell=True)




# Main program execution starts here (when executed from the command line)
if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print usermanual
    else:
        check_invalid_arguments(CLFLAGS.values())
        main()

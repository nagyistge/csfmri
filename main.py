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
# Date: 2017-Jul-29                                                            #
################################################################################

# DESCRIPTION

usermanual = \
    """This program creates 4-dimensional S0 and T2* maps of the brain.

    INPUTS:
           Subject ID
           High-resolution structural image of the brain (nii.gz)
           Functional image (4D) (nii.gz)
           Scanner field map (nii.gz)
           BioPac data file (txt)
    OUTPUTS:
           S0 map (nii.gz)
           T2* map (nii.gz)
    PREREQUISITE:
           FSL 5.0
    
    """

# IMPORTS

import sys
import os
from cl_interface import *
from fsl_interface import *
from csfmri_utils import *

# DEFINITIONS AND CODE

# Command-line arguments
CLFLAGS = {'struct': '-s',
           'func': '-f',
           'fmap': '-m',
           'bio': '-b',
           's0': '-sout',
           't2': '-tout',
           'interactive': '-i'}


# Default output file paths and names
# (directory and extension will be added based on the source file)
# (do not delete any of these, just set to None when needed)
S0_BASE_NAME = "S0"
S0_PATH = "../csfmri_results"    # relative to the source directory
S0_TAG = None
T2_BASE_NAME = "T2"
T2_PATH = "../csfmri_results"    # relative to the source directory
T2_TAG = None


class InputDescriptorObj:
    """An object for passing all input information at once between functions."""
    def __init__(self, StructImgPath, FuncImgPath, FieldMapPath,
                 BioDataPath, S0ImgPath=None, T2ImgPath=None,
                 interactivity=False):
        self.structural = StructImgPath
        self.functional = FuncImgPath
        self.fieldmap = FieldMapPath
        self.biodata = BioDataPath
        self.s0 = S0ImgPath
        self.t2 = T2ImgPath
        self.interactive = interactivity

        # Check integrity of input
        inputcount = {len(self.structural), len(self.functional),
                      len(self.fieldmap), len(self.biodata)}
        if self.s0:
            inputcount.add(len(self.s0))
        if self.t2:
            inputcount.add(len(self.t2))
        assert len(inputcount) == 1, "Integrity of the lists of input " \
                                     "(and output)"
        self.count = inputcount.pop()


def parse_arguments():
    """Subroutine that understands command-line arguments and passes the
    information to the main program."""

    # Read compulsory input arguments: struct, func, fmap and bio
    # Structural image(s)
    StructImgPathList = subarg(CLFLAGS['struct'])
    if not StructImgPathList:
        print ("ERROR: No structural image was specified.")
        exit(1)
    # Functional image(s)
    FuncImgPathList = subarg(CLFLAGS['func'])
    if not FuncImgPathList:
        print ("ERROR: No functional image was specified.")
        exit(1)
    # Field map image(s)
    FieldMapPathList = subarg(CLFLAGS['fmap'])
    if not FieldMapPathList:
        print ("ERROR: No field map image was specified.")
        exit(1)
    # Biometric data
    BioDataPathList = subarg(CLFLAGS['bio'])
    if not BioDataPathList:
        print ("ERROR: No biometric data file was specified.")
        exit(1)

    # Interactivity
    interactivity = argexist(CLFLAGS['interactive'])

    # Validate input
    try:
        StructImgPathList, FuncImgPathList, \
        FieldMapPathList, BioDataPathList = \
            matched_input_validation(
                inputs=[StructImgPathList, FuncImgPathList, FieldMapPathList,
                        BioDataPathList],
                supported_formats=FORMATS.values(),
                interactive=interactivity)
    except:
        raise   # As long as I don't want to override exception handling.

    # Read optional (output) arguments and create them if necessary
    # S0 map
    S0PathList = subarg(CLFLAGS['s0'])
    if not S0PathList:
        print ("WARNING: No output filename was specified for the S0 map. "
               "It will be saved into the source directory with the base name "
               "'{}'".format(os.path.join(S0_PATH, S0_BASE_NAME)))
        try:
            S0PathList = guess_output_from_input(inputfiles=StructImgPathList,
                                                 basename=S0_BASE_NAME,
                                                 tag=S0_TAG, subdir=S0_PATH)
        # FIXME: Placeholder for proper exception handling
        except:
            raise

    # T2* map
    T2PathList = subarg(CLFLAGS['t2'])
    if not T2PathList:
        print ("WARNING: No output filename was specified for the T2 map. "
               "It will be saved into the source directory with the base name "
               "'{}'".format(os.path.join(T2_PATH, T2_BASE_NAME)))
        try:
            T2PathList = guess_output_from_input(inputfiles=StructImgPathList,
                                                 basename=T2_BASE_NAME,
                                                 tag=T2_TAG, subdir=T2_PATH)
        # FIXME: Placeholder for proper exception handling
        except:
            raise

    return InputDescriptorObj(StructImgPathList, FuncImgPathList,
                              FieldMapPathList, BioDataPathList, S0PathList,
                              T2PathList, interactivity)


def summarize(InputObj):
    """Subroutine for printing the summary on the screen."""
    msg = ("\nThe following operation(s) ({}) will be performed:"
           .format(InputObj.count))
    print msg
    print "".join(['='] * len(msg)), "\n"
    for input_group in zip(InputObj.structural, InputObj.functional,
                           InputObj.fieldmap, InputObj.biodata, InputObj):
        print "\n".join(list(input_group)[:-2])
        print "->"
        print "\n".join(list(input_group)[-2:])
        print "\n"


def main():
    """Main program code."""
    # Read and validate input
    InputListObj = parse_arguments()

    # Summarize task and ask for user confirmation
    if InputListObj.interactive:
        summarize(InputObj=InputListObj)
        if not confirmed_to_proceed(msg="Would you like to proceed? (y/n): "):
            exit(1)

    # Perform steps
    fsldir = get_fsldir()


    # Create output

    # Report results


# Main program execution starts here
if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print usermanual
    else:
        # Check for invalid arguments
        invalid_args = check_invalid_arguments(CLFLAGS.values())
        if not invalid_args:
            main()
        else:
            print ("The following argument(s) were not recognised:",
                   " ".join(invalid_args))
            exit(1)

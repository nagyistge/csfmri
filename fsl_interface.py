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

"""This Python module provides Python wrappers for the command-line tools of
FSL. Please note that only part of the full functionality of FSL is covered yet.
"""

import os
from cl_interface import confirmed_to_proceed

def get_fsldir():
    """Returns the path to the FSL installation directory (from $FSLDIR)."""
    fsldir = os.environ['FSLDIR']
    while not os.path.isdir(fsldir):
        print ("FSL is not installed or $FSLDIR variable is set incorrectly.")
        print ("Would you like to set $FSLDIR now? (y/n): ")
        if confirmed_to_proceed():
            fsldir = raw_input("FSLDIR=").lower()
        else:
            exit(1)
    return fsldir

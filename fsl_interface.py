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

"""This Python module provides Python wrappers for the command-line tools of
FSL. Please note that only part of the full functionality of FSL is covered yet.
"""


# IMPORTS

import os
from cl_interface import confirmed_to_proceed
from csfmri_exceptions import NoFSLException


#  DEFINITIONS AND CODE

def get_fsldir():
    """Returns the path to the FSL installation directory (from $FSLDIR). When
    FSL cannot be found, NoFSLException is raised."""
    fsldir = os.environ['FSLDIR']
    while not os.path.isdir(fsldir):
        print ("FSL is not installed or $FSLDIR variable is set incorrectly.")
        print ("Would you like to set $FSLDIR now? (y/n): ")
        if confirmed_to_proceed():
            fsldir = raw_input("FSLDIR=").lower()
        else:
            raise NoFSLException("FSL could not be located on the computer.")
    else:
        # Check if the necessary FSL are really there
        components = {"bet", "fsl_prepare_fieldmap", "fsleyes", "feat",
                      "fsl_anat", "fslsplit", "fslmaths", "fslmerge", "invwarp",
                      "applywarp", "fslchfiletype", "fslcpgeom", "fslroi",
                      "flirt", "mcflirt", "applyxfm4D"}
        if check_fsl_components(fsldir, components):
            return fsldir
        else:
            raise NoFSLException("At least one required FSL tool could not "
                                 "be located on the computer.")


def check_fsl_components(fsldir, components="fsl", bindir="bin"):
    """Checks whether the requested FSL tools can be found in the FSL
    installation directory."""
    if type(components) != set:
        components = set(components)
    tool_exists = [os.path.isfile(os.path.join(fsldir, bindir, comp))
                   for comp in components]
    if all(tool_exists):
        return True
    else:
        return False


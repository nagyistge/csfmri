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

# IMPORTS

import os


# DESCRIPTION

"""This Python module provides classes for handling project-specific exceptions.
"""


# DEFINITIONS AND CODE

class NoFSLException(Exception):
    """This exception is raised when no FSL installation can be located on the
    system."""


class GenericIOException(Exception):
    """This exception is raised in all cases when reading from or writing to
    the disk is unsuccessful."""


class NotFoundException(Exception):
    """This exception is raised when a necessary piece of data or location is
    unavailable."""


class TypeMismatchException(Exception):
    """This exception is raised when two or more objects have incompatible
    types and therefore the required operations are impossible to perform."""


class CountsMismatchException(Exception):
    """This exception is raised when the matched inputs for a given script have
    incompatible counts."""


class MissingArgumentException(Exception):
    """This exception is raised when a compulsory argument is not set or a
    necessary subargument is not given."""


class UnsupportedFormatException(Exception):
    """This exception is raised when a file (usually an input) has a format
    that is not supported by the program."""


class NothingToDoException(Exception):
    """This exception is raised when an expected piece of data (usually input
    file path) is missing and therefore there is nothing to do."""


class NothingDoneException(Exception):
    """This exception is raised when an expected operation could not be
    performed and therefore the results of it will not be available."""


class NIFTIException(Exception):
    """This exception is raised when an image manipulation cannot be carried
    out because the NIfTI data is inaccessible, corrupted or incompatible."""


# Display information if accidentally executed
if __name__ == "__main__":
    print ("{} is not intended for execution."
           .format(os.path.basename(__file__)))
    exit(0)
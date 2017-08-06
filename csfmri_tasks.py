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
# Date: 2017-Aug-06                                                            #
################################################################################

# DESCRIPTION

usermanual = \
    """This Python module contains the modular implementations of all analysis 
    tasks related to the CSFMRI project. This module is not intended for direct 
    execution from the command line."""


# DEVELOPMENT INFO

"""All functions in this module import args, add their task-specific 
information to it and return it to the main process. When the next task is 
executed, the args dictionary already contains information provided by previous 
tasks. This chain-like modular design must be reconsidered for optimising 
performance by parallel execution of certain independent tasks."""


# IMPORTS

import numpy as np
import nibabel as nib
from fsl_interface import get_fsldir
from csfmri_exceptions import *
import subprocess
from collections import OrderedDict


# DEFINITIONS AND CODE

# A comprehensive list of operations that can be performed by the program.
# Based on the program input, boolean values will be added to this set to create
# a dictionary that will instruct the program what to do. Most of the
# functionalities were derived from the original Matlab script as indicated on
# the right.

TASK_ORDER = {'load_fsl': 0,                    # startup.m
             'create_bids_dirs': 1,             # CreateBIDSDirectories.m
             'create_field_map': 2,             # FieldMapBIDS.m
             'load_field_map': 3,               # (added extra functionality)
             'copy_field_map_to_bids': 4,       # First_prepare.m
             'copy_structural_to_bids': 5,      # First_prepare.m
             'copy_single_echo_to_bids': 6,     # First_prepare.m
             'copy_multi_echo_to_bids': 7,      # First_prepare.m
             'create_cheating_ev': 8,           # First_prepare.m
             'pad_single_echo': 9,              # First_prepare.m
             'pad_multi_echo': 10,              # First_prepare.m
             'run_fsl_anat': 11,                # First_prepare.m
             'load_fsl_anatdir': 12,            # (added extra functionality)
             'run_feat': 13,                    # (added extra functionality)
             'load_featdir': 14,                # (added extra functionality)
             'single_echo_analysis': 15,        # Second_GLM.m,
                                                # DualRegressionLoop.m,
                                                # PhaseMapping.m
             'prepare_multi_echo': 16,          # MultiEchoMoCo.m
             'multi_echo_analysis': 17,         # MultiEchoFitAndSort.m
             }

TASK_LIST = set(TASK_ORDER.keys())


def load_fsl(args):
    """Adds the FSL installation path to the program arguments dictionary."""
    print "load_fsl"
    try:
        args['fsldir'] = get_fsldir()
        return args
    except NoFSLException as exc:
        print (exc.message, "Please install FSL before using this program.")
        try:
            subprocess.call(["open", "https://fsl.fmrib.ox.ac.uk/fsl/fslwiki"])
        finally:
            exit(1)


def create_bids_dirs(args):
    print "create_bids_dirs"
    pass


def create_field_map(args):
    print "create_field_map"
    pass


def load_field_map(args):
    print "load_field_map"
    pass


def copy_field_map_to_bids(args):
    print "copy_field_map_to_bids"
    pass


def copy_structural_to_bids(args):
    print "copy_structural_to_bids"
    pass


def copy_single_echo_to_bids(args):
    print "copy_single_echo_to_bids"
    pass


def copy_multi_echo_to_bids(args):
    print "copy_multi_echo_to_bids"
    pass


def create_cheating_ev(args):
    print "create_cheating_ev"
    pass


def pad_single_echo(args):
    print "pad_single_echo"
    pass


def pad_multi_echo(args):
    print "pad_multi_echo"
    pass


def run_fsl_anat(args):
    print "run_fsl_anat"
    pass


def load_fsl_anatdir(args):
    print "load_fsl_anatdir"
    pass


def run_feat(args):
    print "run_feat"
    pass


def load_featdir(args):
    print "load_featdir"
    pass


def single_echo_analysis(args):
    print "single_echo_analysis"
    pass


def prepare_multi_echo(args):
    print "prepare_multi_echo"
    pass


def multi_echo_analysis(args):
    print "multi_echo_analysis"
    pass


# Program execution starts here
if __name__ == "__main__":
    print usermanual
    exit(1)
#!/Volumes/INH_1TB/CSFMRI/venv/venv_csfmri/bin/python

################################################################################
# This script was created part of the CSFMRI (cardio-synchronous fMRI) project
# at the University of Oxford, Centre for Functional Magnetic Resonance Imaging
# of the Brain (FMRIB).
#
# Principal investigator: Professor Peter Jezzard (peter.jezzard@univ.ox.ac.uk)
#
# The original data analysis pipeline was created in Matlab
# by Olivia Viessmann (olivia.viessmann@trinity.ox.ac.uk).
#
# Author: Istvan N. Huszar, M.D. (istvan.huszar@dtc.ox.ac.uk)
# Date: 2017-Jul-29
################################################################################

# IMPORTS

import sys
import os


# DESCRIPTION

"""This Python module provides basic functionality for handling command-line
arguments and user input. It is assumed that all arguments are unique and all
begin with at least one '-'."""


# DEFINITIONS AND CODE

def check_invalid_arguments(valid_args):
    """Returns a list of invalid arguments. An argument is invalid when it is
    not listed in valid_args. If all arguments are valid, a zero-list is
    returned."""
    invalid_args = []
    for arg in sys.argv[1:]:
        if (arg.startswith('-')) and (not (arg in valid_args)):
            invalid_args.append(arg)
    return invalid_args


def argexist(argv, subarg=False):
    """Tells whether a given argument was specified in the command line. When
    the 'subarg' option is True, the argument is expected to have a
    sub-argument. If this is not found, False is returned even if the argument
    itself was specified."""

    if argv in sys.argv:
        if subarg:
            if len(sys.argv)-1 > sys.argv.index(argv):
                if not sys.argv[sys.argv.index(argv)+1].startswith('-'):
                    return True
                else:
                    return False
            else:
                return False
        else:
            return True
    else:
        return False


def subarg(argv, default_value=""):
    """Returns a list of all sub-arguments of a given argument. A default value
    is returned when no sub-argument can be found. The return value is always
    a list of strings. A zero-list is returned if the argument doesn't exist or
    neither sub-arguments nor a default value can be found."""

    arglist = []
    if argexist(argv, subarg=True):
        for arg in sys.argv[sys.argv.index(argv)+1:]:
            if not arg.startswith('-'):
                arglist.append(arg)
            else:
                break
    elif argexist(argv, subarg=False):
        if default_value:
            arglist.append(default_value)
    else:
        pass
    return arglist


def confirmed_to_proceed(msg="", forceanswer=True):
    """Forces user input for continuing the execution of the program."""

    no = {'no', 'n'}
    if forceanswer:
        yes = {'yes', 'y', 'ye'}
    else:
        yes = {'yes', 'y', 'ye', ''}

    choice = raw_input(msg).lower()
    if choice in yes:
        return True
    elif choice in no:
        return False
    else:
        sys.stdout.write("Please respond with 'yes' or 'no': ")
        choice = confirmed_to_proceed()
    return choice


# Display information if accidentally executed
if __name__ == "__main__":
    print ("{} is not intended for execution."
           .format(os.path.basename(__file__)))
    exit(0)
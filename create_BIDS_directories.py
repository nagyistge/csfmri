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
    """This Python module provides a both callable and directly executable
    subroutine that creates a folder structure according to the BIDS
    (Brain Imaging Data Structure) convention.
    
    INPUTS:
        source directory/directories (subject-level)
    OUTPUTS:
        /anat, /fmap, /func directories (customisable)
    
    Usage:
        create_BIDS_directories.py <subject_dirs> [-d <sub-directories>]
    
    """

# IMPORTS

import os
import sys
from cl_interface import *
from csfmri_exceptions import *

# DEFINITIONS AND CODE

CLFLAGS = {'dirs': '-d',
           'interactive': '-i',
           'forced': '-f'}

DEFAULT_DIRS = {'anat', 'fmap', 'func'}


class InputDescriptorObj:
    def __init__(self, SubjectDirs, BIDSDirectories, interactivity=False,
                 forcedmode=False):
        self.subjects = SubjectDirs
        self.BIDSdirs = BIDSDirectories
        self.interactive = interactivity
        self.forcedmode = forcedmode


def CreateBIDSDirectories(subjects, dirs=DEFAULT_DIRS, interactive=False,
                          force=False):
    """Creates the specified BIDS (Brain Imaging Data Structure) directories at
    the specified (subject) location(s). When the interactive option is True,
    all disk operations must be confirmed. The default directories being
    created are defined in DEFAULT_DIRS. In the same instance only the same set
    of directories can be created for all subjects. When 'force' is True, no
    exceptions are raised, only warnings are shown and error sum is returned."""
    # Initialise error sum
    err = 0

    # Create compatibility with vector input.
    if not (type(subjects) is list):
        subjects = [subjects]

    for subject_dir in subjects:
        # Make sure that the source (subject) directory exists, so that further
        # sub-directories can be created.
        while not os.path.isdir(subject_dir):
            if interactive:
                print ("Subject directory '{}' does not exist. Would you like "
                       "to create it now? (y/n): ".format(subject_dir))
                if confirmed_to_proceed():
                    try:
                        os.mkdir(subject_dir)
                        print ("Subject directory '{}' was successfully "
                               "created.".format(subject_dir))
                    except:
                        if force:
                            err = err + 1
                            print ("WARNING: Subject directory '{}' could not "
                                   "be created.".format(subject_dir))
                            break
                        else:
                            raise GenericIOException(
                                "ERROR: Subject directory '{}' could not be "
                                "created.".format(subject_dir))
                else:
                    if force:
                        err = err + 1
                        print ("WARNING: Subject directory '{}' is unavailable."
                               .format(subject_dir))
                        break
                    else:
                        raise NotFoundException("ERROR: Subject directory '{}' "
                                                "is unavailable."
                                                .format(subject_dir))
            else:
                try:
                    os.mkdir(subject_dir)
                    print ("Subject directory '{}' was successfully created."
                           .format(subject_dir))
                except:
                    if force:
                        err = err + 1
                        print ("WARNING: Subject directory '{}' could not be "
                               "created.".format(subject_dir))
                        break
                    else:
                        raise GenericIOException("ERROR: Subject directory '{}'"
                                                 " could not be created."
                                                 .format(subject_dir))

        # Create directories
        dirs = set(dirs)    # avoid duplicates
        current_dir = os.getcwd()
        os.chdir(subject_dir)
        for dirname in dirs:
            try:
                if interactive:
                    print ("CONFIRM: Creating '{}': "
                           .format(os.path.join(subject_dir, dirname)))
                    if confirmed_to_proceed(forceanswer=False):
                        os.mkdir(dirname)
                    else:
                        print ("WARNING: '{}' was not created."
                               .format(os.path.join(subject_dir, dirname)))
                        continue
                else:
                    os.mkdir(dirname)
            except:
                if force:
                    err = err + 1
                    print ("WARNING: The directory '{}' could not be created."
                           .format(os.path.join(subject_dir, dirname)))
                    continue
                raise GenericIOException(
                    "ERROR: The directory '{}' could not be created."
                    .format(os.path.join(subject_dir, dirname)))
            finally:
                # Change back to the original directory
                os.chdir(current_dir)

    # Return error code (=0 if everything went fine)
    return err


def parse_arguments():
    """This sub-routine understands the command-line arguments and passes the
    information to the main code."""
    # Parse subject directory/directories
    SubjectDirs = subarg(sys.argv[0])
    if not SubjectDirs:
        print ("ERROR: No subject directory was specified.")
        exit(1)

    # Whether the specified directories exist will be checked in
    # CreateBIDSDirectories

    # Parse the name of the directories to be created
    # Whether the provide names are valid are not tested in advance
    BIDSDirs = subarg(CLFLAGS['dirs'])
    if not BIDSDirs:
        print ("WARNING: No BIDS names specified. The following directories "
               "will be created by default: {}"
               .format("/"+" /".join(DEFAULT_DIRS)))
        BIDSDirs = list(DEFAULT_DIRS)

    # Parse other options
    # Interactivity
    interactivity = argexist(CLFLAGS['interactive'])

    # Forced mode
    forcedmode = argexist(CLFLAGS['forced'])

    return InputDescriptorObj(SubjectDirs, BIDSDirs, interactivity, forcedmode)


def summarize(InputObj):
    """This sub-routine prints the summary on the screen."""
    msg = "The following directories will be created:"
    print msg
    print "".join(['='] * len(msg)), "\n"
    for subj in InputObj.subjects:
        for BIDS_dirname in InputObj.BIDSdirs:
            print os.path.join(subj, BIDS_dirname)
        print "\n"
    print "Forced mode: {}".format(InputObj.forcedmode)


def main():
    """Main program code. When the module is run from the command line, this
    handles the command-line arguments."""

    # Parse arguments
    InputListObj = parse_arguments()

    # Create summary
    if InputListObj.interactive:
        summarize(InputObj=InputListObj)
        if not confirmed_to_proceed("\nWould you like to proceed? (y/n): "):
            exit(1)

    # Do the job
    CreateBIDSDirectories(InputListObj.subjects, InputListObj.BIDSdirs,
                          InputListObj.interactive, InputListObj.forcedmode)


# Main program execution starts here (when executed from the command line)
if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print usermanual
    else:
        check_invalid_arguments(CLFLAGS.values())
        main()

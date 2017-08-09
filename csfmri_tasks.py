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

"""All public functions in this module import args, add their task-specific 
information to it and return it to the main process. When the next task is 
executed, the args dictionary already contains information provided by previous 
tasks. This chain-like modular design must be reconsidered for optimising 
performance by parallel execution of certain independent tasks.

The module contains a few private functions; these are indicated with a leading 
single underscore."""


# IMPORTS

import numpy as np
import nibabel as nib
from cl_interface import confirmed_to_proceed
from fsl_interface import get_fsldir
from csfmri_exceptions import *
import subprocess
import datetime
import os
from errno import EEXIST
import shutil
import dask.array as da


# DEFINITIONS AND CODE

# A comprehensive list of operations that can be performed by the program.
# Based on the program input, boolean values will be added to this set to create
# a dictionary that will instruct the program what to do. Most of the
# functionalities were derived from the original Matlab script as indicated on
# the right.

TASK_ORDER = {'load_fsl': 0,                     # startup.m
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

# Default BIDS directories
FMAP_DIR = "fmap"   # for field map and related images
ANAT_DIR = "anat"   # for structural image (not the output of fsl_anat!)
FUNC_DIR = "func"   # for the functional image(s) and reference image(s)

# Default file name tags (the prefix is the subjectID stored in args['label'])
# FIXME: Hard coding the extension is bad practice.
ANAT_TAG = "_structural.nii.gz"
FMAP_TAG = "_fmap.nii.gz"
FMAG_TAG = "_fmap_magnitude.nii.gz"
FPHASE_TAG = "_fmap_phasediff.nii.gz"
FMAG_BET_TAG = "_fmap_magnitude_brain.nii.gz"
SECHO_TAG = "_task_rest.nii.gz"
SREF_TAG = "_task_rest_sbref.nii.gz"
MECHO_TAG = "_task_rest_multiecho.nii.gz"
MREF_TAG = "_task_rest_multiecho_sbref.nii.gz"
STRUCT_BASE_NAME = "structural"

# Other built-in constants
N_PAD_SLICES = 2

def extract_subject_label(args):
    """Extracts the subject ID (label) from the 'id' parameter that is
    potentially specified as a folder path."""
    if args['id'].endswith(os.path.sep):
        args['id'] = args['id'][:1]
    label = os.path.split(args['id'])[-1]

    return label


def absolutise_paths(args):
    """Convert all paths to absolute paths. All relative paths should be
    referenced to the current working directory."""
    path_args = {'id', 'struct', 'anatdir', 'single_echo', 'sref', 'stime',
                 'sfeat', 'sbio', 'multi_echo', 'mref', 'mtime', 'mbio', 'fmap',
                 'fmag', 'fphase', 'log', 'config'}
    for key in path_args:
        if args[key] is not None:
            args[key] = os.path.abspath(args[key])

    return args


def validate_input_paths(args):
    """Checks whether the provided paths for input arguments are valid, i.e.
    they represent a file or directory. If an invalid path is found, the input
    argument is set to None."""
    file_path_args = {'struct', 'single_echo', 'sref', 'stime', 'sbio',
                      'multi_echo', 'mref', 'mtime', 'mbio', 'fmap',
                      'fmag', 'fphase', 'log', 'config'}
    dir_path_args = {'anatdir', 'sfeat'}

    for key in file_path_args:
        if not os.path.isfile(args[key]):
            args[key] = None
    for key in dir_path_args:
        if not os.path.isdir(args[key]):
            args[key] = None

    return args


def _status(msg, args):
    """Displays msg on screen in Verbose Mode. Saves msg to log file, if
    logging to file is enabled. In case of an I/O exception, logging will be
    automatically turned off for the current session and a warning will be
    displayed."""

    msg = '[' + str(datetime.datetime.now()).split('.')[0] + '] ' + msg

    # On-screen logging
    if args['verbose']:
        print (msg)
    else:
        pass

    # If log file path doesn't exist, create it.
    logdir = os.path.split(args['log'])[0]
    if not os.path.isdir(logdir):
        try:
            _mkdir_p(logdir)
            _status("Log directory with log file was created successfully at {}"
                    .format(args['log']), args)
        except:
            print ('[' + str(datetime.datetime.now()).split('.')[0] + '] ' +
                   "ERROR: Log file could not be accessed at {}. "
                   "Logging was disabled.".format(args['log']))
            args['log'] = False

    # Logging into file
    if args['log']:
        try:
            with open(args['log'], "a+") as logfile:
                logfile.write(msg+"\n")
        except IOError:
            print ('[' + str(datetime.datetime.now()).split('.')[0] + '] ' +
                   "ERROR: Log file could not be accessed at {}. "
                   "Logging was disabled.".format(args['log']))
            args['log'] = False
    else:
        pass


def _secure_path(dirpath, args, confirm_message=None, msg_success=None,
                 msg_failure=None):
    """This sub-routine provides a template for the user-friendly handling of
    non-existent directories on the go. This routine writes directly on the
    screen and into the log file of the actual session. Raises
    GenericIOException when the directory cannot be created."""

    # Check if the directory exists
    if os.path.isdir(dirpath):
        pass
    else:
        # Log the problem
        _status("The directory does not seem to exist at '{}'.".format(dirpath),
                args)
        # Create messages
        if confirm_message is None:
            confirm_message = "Would you like to create it now? (y/n): "\
                              .format(dirpath)
        if msg_success is None:
            msg_success = "{} was successfully created.".format(dirpath)
        if msg_failure is None:
            msg_failure = "{} could not be created.".format(dirpath)

        # Let the user create the directory in Interactive Mode.
        if not args['auto']:
            if confirmed_to_proceed(confirm_message):
                try:
                    _mkdir_p(dirpath)
                except:
                    _status(msg_failure, args)
                    raise GenericIOException()
            # If the user decides not to create the directory.
            else:
                _status(msg_failure, args)
                raise GenericIOException()
        # Try creating the directory automatically when Interactive Mode
        # is switched off.
        else:
            try:
                _mkdir_p(dirpath)
            except:
                _status(msg_failure, args)
                raise GenericIOException()


def _run(command, args, bg=False):
    """Wrapper that uses the subprocess module to run execute a shell command.
    With the bg option being True, the child process will be executed in the
    background. Logs stdout and stderr according to the session preferences,
    when the run has been completed."""
    _status("COMMAND ISSUED: '" + " ".join(command) + "'", args)

    if not bg:
        try:
            procid = subprocess.Popen(command, stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE)
            stdout, stderr = procid.communicate()
            _status("Command output: \n{}\nCommand errors: {}"
                    .format(str(stdout), str(stderr)), args)
        except:
            msg = "ERROR while issuing the command: '{}'"\
                  .format(command)
            _status(msg, args)
            raise NothingDoneException(msg)
    else:
        try:
            procid = subprocess.Popen(command)
            stdout = procid.stdout.read()
            stderr = procid.stderr.read()
            _status("Command output: \n{}\nCommand errors: {}"
                    .format(str(stdout), str(stderr)), args)
        except:
            msg = "ERROR while issuing the command: '{}'" \
                .format(command)
            _status(msg, args)
            raise NothingDoneException(msg)


def _copy(source, dest, args, key=None, description="file", msg_notfound=None,
          msg_success=None, msg_failure=None):
    """Template for the highly stereotyped steps associated with copying a file
    from one location to another. These include testing if the source file is
    valid and the destination directory exists, as well as handling I/O
    exceptions and setting the respective argument descriptor to the new value.
    """

    # Create messages
    if msg_notfound is None:
        msg_notfound = "No {} was found at {}."\
            .format(str(description).lower(), source)
    if msg_success is None:
        msg_success = "The {} was successfully copied from {} " \
                      "to {}.".format(str(description).lower(), source, dest)
    if msg_failure is None:
        msg_failure = "The {} could not be copied from {} to {}"\
                      .format(str(description).lower(), source, dest)

    # Update status
    _status("Copying the {}...".format(str(description).lower()), args)

    # Check if the source file exists
    if not os.path.isfile(source):
        _status(msg_notfound, args)
    else:
        # Check if the destination directory exists
        targetdir, fname = os.path.split(dest)
        try:
            _secure_path(targetdir, args)
        except GenericIOException:
            # Set the destination directory to the subject directory
            # (universally).
            targetdir = args['id']
            _status("The {} will be copied from '{}' to the subject directory: "
                    "'{}'".format(str(description).lower(), source, targetdir),
                    args)

        # Copy the file
        try:
            new_dest = os.path.join(targetdir, fname)
            shutil.copy(source, new_dest)
            _status(msg_success, args)

            # Update the information about the object in the program argument
            # dictionary
            if key is not None:
                args[key] = new_dest
                _status("Path to the {} was set to {}"
                        .format(str(description).lower(), new_dest), args)
        except:
            # FIXME: Add exception handling
            _status(msg_failure, args)
            raise GenericIOException(msg_failure)


def load_fsl(args):
    """Adds the FSL installation path to the program arguments dictionary."""
    try:
        args['fsldir'] = get_fsldir()
        _status("FSL installation was found at '{}'".format(args['fsldir']),
                args)
    except NoFSLException as exc:
        _status(exc.message + " Please (re)install FSL before using this "
                              "program.", args)
        try:
            subprocess.call(["open", "https://fsl.fmrib.ox.ac.uk/fsl/fslwiki"])
        finally:
            exit(1)


# Source: https://stackoverflow.com/questions/600268/
# mkdir-p-functionality-in-python
def _mkdir_p(path):
    """Creates directories along the entire specified path. The existing
    directory exception is suppressed."""
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def create_bids_dirs(args):
    """Creates the specified BIDS (Brain Imaging Data Structure) directories at
    the specified (subject) location. When the interactive option is True,
    all disk operations must be confirmed."""

    _status("Creating BIDS directory tree...", args)
    # Make sure that the subject directory exists, so that further
    # sub-directories can be created. If the requested directory is not
    # accessible, set subject directory to the current working directory.
    try:
        _secure_path(args['id'], args)
    except GenericIOException:
        args['id'] = os.getcwd()
        _status("Subject directory was set to the current working directory: "
                "'{}'".format(args['id']), args)

    # Create directories
    # Avoid duplicates among BIDS directories
    bids_dirs = set(args['bids_dirs'])
    for dirname in bids_dirs:
        try:
            dirname = os.path.abspath(os.path.join(args['id'], dirname))
            if not args['auto']:
                if confirmed_to_proceed("CONFIRM: Creating '{}': "
                   .format(os.path.join(args['id'], dirname)),
                                        forceanswer=False):
                    _mkdir_p(dirname)
                else:
                    _status("WARNING: '{}' was not created."
                           .format(os.path.join(args['id'], dirname)), args)
                    continue
            else:
                _mkdir_p(dirname)
        except:
            # FIXME: Add exception handling
            _status("ERROR: The directory '{}' could not be created."
                    .format(os.path.join(args['id'], dirname)), args)
    else:
        _status("The BIDS directory tree was successfully created.", args)


def create_field_map(args):
    """Calculates scanner field map from magnitude and phase difference images.
    The output is saved into the subject directory or copied to the respective
    BIDS sub-directory (/fmap) when -copy is set."""

    _status("Calculating field map...", args)
    if args['copy']:
        # Check if the standard BIDS directory exist
        try:
            targetdir = os.path.join(args['id'], FMAP_DIR)
            confirm_msg = "Would you like to create the standard directory " \
                          "for field maps now? (y/n): "
            _secure_path(targetdir, args, confirm_message=confirm_msg)
        except GenericIOException:
            targetdir = args['id']
            _status("The field map will be saved into the subject directory: "
                    "'{}'.".format(targetdir), args)

        # Copy and rename the magnitude image.
        try:
            new_fmag = os.path.join(targetdir, args['label'] + FMAG_TAG)
            shutil.copy(args['fmag'], new_fmag)
            _status("Field map magnitude image was successfully copied from "
                    "'{}' to '{}'.".format(args['fmag'], new_fmag), args)
            # Update information about field map magnitude image in the program
            # argument dictionary
            args['fmag'] = new_fmag
            _status("Field map magnitude image was set to '{}'."
                    .format(args['fmag'], new_fmag), args)
        except:
            # FIXME: Add exception handling
            _status("Field map magnitude image could not be copied from '{}' "
                    "to '{}'".format(args['fmag'], new_fmag), args)

        # Copy and rename the phase difference image.
        try:
            new_fphase = os.path.join(targetdir, args['label'] + FPHASE_TAG)
            shutil.copy(args['fphase'], new_fphase)
            _status("Field map phase difference image was successfully copied "
                    "from '{}' to '{}'.".format(args['fphase'], new_fphase),
                    args)
            # Update information about field map phase difference image in the
            # program argument dictionary
            args['fphase'] = new_fphase
            _status("Field map phase image was set to '{}'."
                    .format(args['fphase'], new_fphase), args)
        except:
            # FIXME: Add exception handling
            _status("Field map phase image could not be copied from '{}' "
                    "to '{}'".format(args['fphase'], new_fphase), args)
    else:
        pass

    # Run brain extraction on magnitude image (by calling bet from FSL)
    betcmd = [os.path.join(args['fsldir'], "bin/bet"), new_fmag,
              os.path.join(targetdir, args['label']) +
              FMAG_BET_TAG, '-f', str(args['fractint'])]
    try:
        _run(betcmd, args, bg=False)
    except NothingDoneException:
        # FIXME: Add exception handling
        raise

    # Prepare the field map (by calling fsl_prepare_fieldmap)
    fpfcmd = [os.path.join(args['fsldir'], "bin/fsl_prepare_fieldmap"),
              'SIEMENS', new_fphase,
              os.path.join(targetdir, str(args['label'])) + FMAG_BET_TAG,
              os.path.join(targetdir, str(args['label'])) + FMAP_TAG,
              str(args['echodiff'])]
    try:
        _run(fpfcmd, args, bg=False)
    except NothingDoneException:
        # FIXME: Add exception handling
        raise

    # Update field map information in the program argument dictionary
    args['fmap'] = os.path.join(targetdir, str(args['label'])) + FMAP_TAG
    _status("Path to field map was set to {}".format(args['fmap']), args)


def load_field_map(args):
    """Creates reference to an existing scanner field map. If the copy argument
    is set, the file is copied to its standard BIDS sub-directory (/fmap)."""

    # Check if the field map exists.
    if not os.path.isfile(args['fmap']):
        _status("The field map could not be loaded from {}"
                .format(args['fmap']), args)
    else:
        # Check if it is necessary to copy and rename the file
        if not args['copy']:
            _status("Field map was loaded from {}".format(args['fmap']), args)
        else:
            # Specify target directory (this is a built-in constant)
            targetdir = os.path.join(args['id'], FMAP_DIR)
            new_fmap = os.path.join(targetdir, args['label'] + FMAP_TAG)
            if os.path.samefile(args['fmap'], new_fmap):
                _status("Field map was loaded from {}".format(args['fmap']),
                        args)
                args['fmap'] = new_fmap
                _status("Path to field map was set to {}".format(args['fmap']),
                        args)
            else:
                # Check if the standard BIDS directory exists
                try:
                    _secure_path(targetdir, args)
                except GenericIOException:
                    targetdir = args['id']
                    _status("Field map will be copied from '{}' to the subject "
                            "directory: '{}'".format(args['fmap'], targetdir),
                            args)

                try:
                    # Copy the file
                    shutil.copy(args['fmap'], new_fmap)
                    _status("The field map was successfully copied from {} to "
                            "{}.".format(args['fmap'], new_fmap), args)

                    # Update the field map information in the program argument
                    # dictionary
                    args['fmap'] = new_fmap
                    _status("Path to field map was set to {}"
                            .format(args['fmap']), args)
                except:
                    # FIXME: Add exception handling
                    _status("The field map could not be copied from {} to {}."
                            .format(args['fmap'], new_fmap), args)


def copy_structural_to_bids(args):
    """Copy structural scan into the respective BIDS sub-directory (/anat)."""

    # Check if the argument of 'copy' is indeed True. (Although it is impossible
    # to set it to False through the command line, the config file carries this
    # possibility.)
    if not args['copy']:
        # Stay idle if the copy argument was indeed set but set to False.
        pass
    else:

        # Specify target directory (this is a built-in constant)
        targetdir = os.path.join(args['id'], ANAT_DIR)
        # Check if the file given by the argument indeed exists
        if not os.path.isfile(args['struct']):
            _status("There is no structural image at {}."
                    .format(args['struct']), args)
        else:
            # Check if the standard BIDS directory exists
            try:
                _secure_path(targetdir, args)
            except GenericIOException:
                targetdir = args['id']
                _status("Structural image will be copied from '{}' to the "
                        "subject directory: '{}'"
                        .format(args['struct'], targetdir), args)

            # Copy the file
            try:
                new_struct = os.path.join(targetdir, args['label'] + ANAT_TAG)
                shutil.copy(args['struct'], new_struct)
                _status("The structural image was successfully copied from {} "
                        "to {}.".format(args['fmap'], new_struct), args)

                # Update the information about the location of the structural image
                # in the program argument dictionary
                args['struct'] = new_struct
                _status("Structural image path was set to {}"
                        .format(new_struct), args)
            except:
                # FIXME: Add exception handling
                _status("The structural image could not be copied from {} to {}"
                        .format(args['fmap'], new_struct), args)


def copy_single_echo_to_bids(args):
    """Copy single-echo functional scan and single-echo reference image into the
     respective BIDS sub-directory (/func)."""

    # Check if the argument of 'copy' is indeed True. (Although it is impossible
    # to set it to False through the command line, the config file carries this
    # possibility.)
    if not args['copy']:
        # Stay idle if the copy argument was indeed set but set to False.
        pass
    else:
        # Specify target directory (this is a built-in constant)
        targetdir = os.path.join(args['id'], FUNC_DIR)

        # Copy the single-echo functional image
        if os.path.isfile(args['single_echo']):
            try:
                new_secho = os.path.join(targetdir, args['label'] + SECHO_TAG)
                _copy(args['single_echo'], new_secho, args, key='single_echo',
                      description="single-echo functional image")
            except:
                # TODO: Add exception handling
                # Try to do as much as possible, so ignore the failure for now.
                pass

        # Copy the single-echo reference image
        if os.path.isfile(args['sref']):
            try:
                new_sref = os.path.join(targetdir, args['label'] + SREF_TAG)
                _copy(args['sref'], new_sref, args, key='sref',
                      description="single-echo reference image")
            except:
                # TODO: Add exception handling
                # Try to do as much as possible, so ignore the failure for now.
                pass


def copy_multi_echo_to_bids(args):
    """Copy multi-echo functional scan and multi-echo reference image into the
     respective BIDS sub-directory (/func)."""

    # Check if the argument of 'copy' is indeed True. (Although it is impossible
    # to set it to False through the command line, the config file carries this
    # possibility.)
    if not args['copy']:
        # Stay idle if the copy argument was indeed set but set to False.
        pass
    else:
        # Specify target directory (this is a built-in constant)
        targetdir = os.path.join(args['id'], FUNC_DIR)

        # Copy the multi-echo functional image
        if os.path.isfile(args['multi_echo']):
            try:
                new_mecho = os.path.join(targetdir, args['label'] + MECHO_TAG)
                _copy(args['multi_echo'], new_mecho, args, key='multi_echo',
                      description="multi-echo functional image")
            except:
                # TODO: Add exception handling
                # Try to do as much as possible, so ignore the failure for now.
                pass

        # Copy the multi-echo reference image
        if os.path.isfile(args['mref']):
            try:
                new_mref = os.path.join(targetdir, args['label'] + MREF_TAG)
                _copy(args['mref'], new_mref, args, key='mref',
                      description="multi-echo reference image")
            except:
                # TODO: Add exception handling
                # Try to do as much as possible, so ignore the failure for now.
                pass


def create_cheating_ev(args):
    """Creates text file for each of the functional images. It contains the
    values of a mock explanatory variable, so that it can be imported into
    FEAT. The first value is one, all the rest are zeros. The
    length of the EV is equal to the number of volumes in the respective
    functional image. The files are saved into the subject directory."""

    _status("Creating the cheating explanatory variable for the single-echo "
            "image...", args)
    if not args['single_echo']:
        msg = "ERROR: Single-echo functional image was not specified."
        _status(msg, args)
        raise NotFoundException(msg)
    else:
        try:
            vols = nib.load(args['single_echo']).header.get_data_shape()[3]
        except:
            msg = "Single-echo image could not be opened from '{}'."\
                  .format(args['single_echo'])
            _status(msg, args)
            raise NIFTIException(msg)
        fname = os.path.join(args['id'], args['label'] +
                             "_CheatingEV_SingleEcho_{}Vols.txt".format(vols))
        ev = np.zeros((vols, 1), dtype=np.int8)
        ev[0, 0] = 1
        try:
            np.savetxt(fname, ev.astype(np.int8), fmt="%d")
            _status("The cheating EV file was successfully saved to '{}'."
                    .format(fname), args)
        except:
            msg = "ERROR: The cheating EV file could not be saved to '{}'."\
                  .format(fname)
            _status(msg, args)
            raise GenericIOException(msg)

    # MULTI-ECHO (Not used at the moment)
    _status("Creating the cheating explanatory variable for the multi-echo "
            "image...", args)
    if not args['multi_echo']:
        msg = "ERROR: Multi-echo functional image was not specified."
        _status(msg, args)
        raise NotFoundException(msg)
    else:
        try:
            vols = nib.load(args['multi_echo']).header.get_data_shape()[3]
        except:
            msg = "Multi-echo image could not be opened from '{}'."\
                  .format(args['multi_echo'])
            _status(msg, args)
            raise NIFTIException(msg)
        fname = os.path.join(args['id'], args['label'] +
                             "_CheatingEV_MultiEcho_{}Vols.txt".format(vols))
        ev = np.zeros((vols, 1), dtype=np.int8)
        ev[0, 0] = 1
        try:
            np.savetxt(fname, ev.astype(np.int8), fmt="%d")
            _status("The cheating EV file was successfully created at '{}'."
                    .format(fname), args)
        except:
            msg = "ERROR: The cheating EV file could not be created at '{}'."\
                  .format(fname)
            _status(msg, args)
            raise GenericIOException(msg)


def _pad(imgpath, n_slices=2):
    """Pads NIfTI volume from both edges with n zero-slices along the z axis."""

    # Load the NIfTI volume
    img = nib.load(imgpath)
    # Adjust header information
    hdr = img.header
    img_shape = list(hdr.get_data_shape())
    img_shape[2] += 2 * n_slices
    hdr.set_data_shape(img_shape)
    # Manipulate image content
    img = img.get_data()
    zero_shape = img_shape
    zero_shape[2] = 2
    zeroslices = np.zeros(img_shape)
    # Use the dask library for parallel array concatenation
    img = da.concatenate([zeroslices, img, zeroslices], axis=2)

    # Return NIfTI image
    return nib.Nifti1Image(img, hdr.get_sform(), hdr)


def pad_single_echo(args):
    """Adds a number (N_PAD_SLICES) of zero-slices to both ends of the
    single-echo images along the z axis. This is for compatibility with the
    motion correction algorithm in the FSL."""

    # Single-echo functional image
    _status("Padding single-echo functional image with {} zero-slices on both "
            "ends along the z axis...".format(N_PAD_SLICES), args)
    if not args['single_echo']:
        msg = "ERROR: Single-echo functional image was not specified."
        _status(msg, args)
        raise NotFoundException(msg)
    else:
        padded_nifti = _pad(args['single_echo'], n_slices=N_PAD_SLICES)
        # Save it next to the single_echo image
        # FIXME: Hard coding the extension is bad practice.
        fpath, fname = os.path.split(args['single_echo'])
        fname = fname.replace(".nii.gz", "_padded{}.nii.gz"
                              .format(N_PAD_SLICES))
        fname = os.path.join(fpath, fname)
        try:
            nib.save(padded_nifti, fname)
            _status("Padded single-echo functional image was saved to '{}'"
                    .format(fname), args)
        except:
            # Supress error and try to do as many tasks as possible.
            msg = "ERROR while saving the padded single-echo functional " \
                  "image to '{}'.".format(fname)
            _status(msg, args)
            pass

    # Single-echo reference image
    _status("Padding single-echo reference image with {} zero-slices "
            "on both ends along the z axis...".format(N_PAD_SLICES), args)
    if not args['sref']:
        msg = "ERROR: Single-echo reference image was not specified."
        _status(msg, args)
        raise NotFoundException(msg)
    else:
        padded_nifti = _pad(args['sref'], n_slices=N_PAD_SLICES)
        # Save it next to the single_echo image
        # FIXME: Hard coding the extension is bad practice.
        fpath, fname = os.path.split(args['sref'])
        fname = fname.replace(".nii.gz", "_padded{}.nii.gz"
                              .format(N_PAD_SLICES))
        fname = os.path.join(fpath, fname)
        try:
            nib.save(padded_nifti, fname)
            _status("Padded single-echo reference image was saved to '{}'"
                    .format(fname), args)
        except:
            # Supress error and try to do as many tasks as possible.
            msg = "ERROR while saving the padded single-echo reference " \
                  "image to '{}'.".format(fname)
            _status(msg, args)
            pass


def pad_multi_echo(args):
    """Adds a number (N_PAD_SLICES) of zero-slices to both ends of the
    multi-echo images along the z axis. This is for compatibility with the
    motion correction algorithm in the FSL."""

    # Multi-echo functional image
    _status("Padding multi-echo functional image with {} zero-slices on both "
            "ends along the z axis...".format(N_PAD_SLICES), args)
    if not args['multi_echo']:
        msg = "ERROR: Multi-echo functional image was not specified."
        _status(msg, args)
        raise NotFoundException(msg)
    else:
        padded_nifti = _pad(args['multi_echo'], n_slices=N_PAD_SLICES)
        # Save it next to the single_echo image
        # FIXME: Hard coding the extension is bad practice.
        fpath, fname = os.path.split(args['multi_echo'])
        fname = fname.replace(".nii.gz", "_padded{}.nii.gz"
                              .format(N_PAD_SLICES))
        fname = os.path.join(fpath, fname)
        try:
            nib.save(padded_nifti, fname)
            _status("Padded multi-echo functional image was saved to '{}'"
                    .format(fname), args)
        except:
            # Supress error and try to do as many tasks as possible.
            msg = "ERROR while saving the padded multi-echo functional " \
                  "image to '{}'.".format(fname)
            _status(msg, args)
            pass

    # Single-echo reference image
    _status("Padding multi-echo reference image with {} zero-slices "
            "on both ends along the z axis...".format(N_PAD_SLICES), args)
    if not args['mref']:
        msg = "ERROR: Multi-echo reference image was not specified."
        _status(msg, args)
        raise NotFoundException(msg)
    else:
        padded_nifti = _pad(args['mref'], n_slices=N_PAD_SLICES)
        # Save it next to the single_echo image
        # FIXME: Hard coding the extension is bad practice.
        fpath, fname = os.path.split(args['mref'])
        fname = fname.replace(".nii.gz", "_padded{}.nii.gz"
                              .format(N_PAD_SLICES))
        fname = os.path.join(fpath, fname)
        try:
            nib.save(padded_nifti, fname)
            _status("Padded multi-echo reference image was saved to '{}'"
                    .format(fname), args)
        except:
            # Supress error and try to do as many tasks as possible.
            msg = "ERROR while saving the padded multi-echo reference " \
                  "image to '{}'.".format(fname)
            _status(msg, args)
            pass


def run_fsl_anat(args):
    """Calls fsl_anat to perform re-orientation, bias-field correction, and
    tissue-type segmentation on the structural image."""

    _status("Running fsl_anat on structural image...", args)
    if not args['struct']:
        msg = "Structural image path is not specified."
        _status(msg, args)
        raise NotFoundException(msg)
    else:
        if not args['fsldir']:
            args['fsldir'] = get_fsldir()
        anatcmd = [os.path.join(args['fsldir'], "bin/fsl_anat"),
                   "--nosubcortseg", "-i", args['struct'], "-o",
                   os.path.join(args['id'], STRUCT_BASE_NAME)]
        try:
            _run(anatcmd, args, bg=False)
            # Update the fsl_anat directory information in the program argument
            # dictionary
            new_anat = os.path.join(args['id'], STRUCT_BASE_NAME +
                                           ".anat")
            if os.path.isdir(new_anat):
                args['anatdir'] = new_anat
                _status("Path to the .anat directory was set to {}"
                        .format(new_anat), args)
            else:
                _status("ERROR: Path to the .anat directory could not be set "
                        "after running fsl_anat.", args)
        except NothingDoneException as exc:
            # FIXME: Add exception handling
            raise


def load_fsl_anatdir(args):
    """Loads existing output directory of a previous fsl_anat session."""

    if not os.path.isdir(args['anatdir']):
        _status("The existing fsl_anat directory could not be loaded from '{}'."
                .format(args['anatdir']), args)
    else:
        _status("Path to the .anat directory was set to {}"
                .format(args['anatdir']), args)
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
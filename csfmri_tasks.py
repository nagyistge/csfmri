#!/usr/bin/env python

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
from math import ceil
from cardioresp_GLM import GLMObject
import glob
from sklearn.cluster import KMeans
import peakutils
import copy


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
              'load_biodata': 15,
              'single_echo_analysis': 16,        # Second_GLM.m,
                                                 # DualRegressionLoop.m,
                                                 # PhaseMapping.m
              'prepare_multi_echo': 17,          # MultiEchoMoCo.m
              'multi_echo_analysis': 18,         # MultiEchoFitAndSort.m
              }

TASK_LIST = set(TASK_ORDER.keys())

# Default BIDS directories
FMAP_DIR = "fmap"   # for field map and related images
ANAT_DIR = "anat"   # for structural image (not the output of fsl_anat!)
FUNC_DIR = "func"   # for the functional image(s) and reference image(s)
FEAT_DIR = "singleecho"  # base name for feat directory
MASK_DIR = "masks"  # for the eroded brain masks (single-echo analysis)
BIO_DIR = "bio"     # for the biopac recordings
RESULTS_DIR = "results"     # output directory for single-echo and multi-echo
#                             analyses

# Default file name tags (the prefix is the subjectID stored in args['label'])
# FIXME: Hard coding the extension is bad practice.
ANAT_TAG = "_structural.nii.gz"
FMAP_TAG = "_fmap.nii.gz"
FMAG_TAG = "_fmap_magnitude.nii.gz"
FPHASE_TAG = "_fmap_phasediff.nii.gz"
SECHO_TAG = "_task_rest.nii.gz"
SREF_TAG = "_task_rest_sbref.nii.gz"
STIME_TAG = "_single_echo_timing.txt"
MECHO_TAG = "_task_rest_multiecho.nii.gz"
MREF_TAG = "_task_rest_multiecho_sbref.nii.gz"
MTIME_TAG = "_multi_echo_timing.txt"
STRUCT_BASE_NAME = "structural"
ERO_NAME = "ero_mask.nii.gz"
SBIO_TAG = "_biopac_single_echo.txt"
MBIO_TAG = "_biopac_multi_echo.txt"
CARDMAP_TAG = "_cardmap.nii.gz"
RESPMAP_TAG = "_respmap.nii.gz"
PHASEMAP_TAG = "_phasemap.nii.gz"

# Other built-in constants
# Number of zero-slices used to pad each z-end of the NIfTI volumes
N_PAD_SLICES = 2
# Number of consecutive erosion steps (single-echo analysis)
N_EROSIONS = 3
# Column order (left to right) in the physiological data file
# 'sats' is not used, but the other three are compulsory.
BIOFILE_COLUMN_ORDER = {'respiratory':  0,
                        'trigger':      1,
                        'cardiac':      2,
                        'sats':         3}
# Trigger pulse width
TRIGGER_DURATION = 70   # ms


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
            args[key] = os.path.realpath(os.path.abspath(args[key]))

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
                    _status(msg_success, args)
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
                _status(msg_success, args)
            except:
                _status(msg_failure, args)
                raise GenericIOException()

    return dirpath


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
            procid = subprocess.Popen(command, stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE)
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

        # Avoid overwriting itself
        new_dest = os.path.join(targetdir, fname)
        if os.path.realpath(source) == new_dest:
            pass
        else:
            # Copy the file
            try:
                shutil.copy2(source, new_dest)
                _status(msg_success, args)
            except:
                # FIXME: Add exception handling
                _status(msg_failure, args)
                raise GenericIOException(msg_failure)

        # Update the information about the object in the program argument
        # dictionary
        if key is not None:
            args[key] = new_dest
            _status("Path to the {} was set to {}"
                    .format(str(description).lower(), new_dest), args)


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


def _parse_timing_file(timing_file, args):
    """Returns a dictionary filled with the timing information stored in a
    timing descriptor file. The values of the dictionary are float, bool or
    numpy array, respectively."""

    # Update status
    _status("Parsing timing descriptor file at '{}'...".format(timing_file),
            args)

    if not os.path.isfile(timing_file):
        msg = "ERROR: The provided timing descriptor file does not exist."
        _status(msg, args)
        raise NothingDoneException(msg)
    else:
        # Read all lines
        try:
            with open(timing_file, mode="r") as f:
                lines = f.readlines()
        except:
            raise IOError("ERROR: Timing descriptor file could not be opened "
                          "from '{}'.".format(timing_file))

        # Remove blank lines
        lines = [line for line in lines if line]

        # Concatenate lines that end with a line continuation character (\)
        # Note that the \ character is considered as a line continuation only
        # at the end of each line if and only if there is whitespace on both
        # sides of it.
        i = 0
        while i < len(lines) - 1:
            # Loop until the current line contains a suspect line cont. char.
            # This is necessary for multi-line continuation.
            while not lines[i].split("\\")[-1].strip():
                line = lines[i].split("\\")[:-1]
                # Check whitespace from the left
                if line[-1].endswith((" ", "\t")):
                    line_end = [lines[i + 1]]
                    lines[i] = "".join(line + line_end)
                    lines.remove(lines[i + 1])
                else:
                    # If no whitespace on the left, discard and move to the next
                    # line.
                    break
            i += 1
        # Discard accidental line continuation in the last line
        lines[-1] = lines[-1].replace("\\\n", "\n")

        # Discard comment lines and lines without assignment
        lines = [line for line in lines
                 if (not line.startswith("#")) and (line.find("=") != -1)]

        # Initialise a dictionary with all timing attributes
        timing = {"TR": None, "SS": None, "TE": None, "ST": None,
                  "PADDED": None}

        for line in lines:
            # Discard whitespace (space, tab)
            line = line.replace(" ", "")
            line = line.replace("\t", "")

            # Discard in-line comment and newline character
            line = line.split("#")[0].strip()

            # Import timing data into a dictionary
            argname = line.split("=")[0]
            argval = "=".join(line.split("=")[1:]).split(",")

            if timing[argname] is not None:
                _status("WARNING: Argument '{}' specified more than once. The "
                        "last specification will be used."
                        .format(argname), args)

            if argval != ['']:
                if argname == "TR":
                    timing[argname] = [float(val) for val in argval
                                       if val != ""]
                    if len(timing[argname]) != 1:
                        _status("WARNING: TR can take a single value. Only the "
                                "first specified value was considered.", args)
                    timing[argname] = timing[argname][0]
                elif argname == "SS":
                    timing[argname] = [float(val) for val in argval
                                       if val != ""]
                    if len(timing[argname]) != 1:
                        _status("WARNING: SS can take a single value. Only the "
                                "first specified value was considered.", args)
                    timing[argname] = timing[argname][0]
                elif argname == "TE":
                    timing[argname] = np.array([float(val) for val in argval
                                                if val != ""])
                elif argname == "ST":
                    timing[argname] = np.array([float(val) for val in argval
                                                if val != ""])
                elif argname == "PADDED":
                    timing[argname] = [eval(str(val).title())
                                       for val in argval if val != ""]
                    if len(timing[argname]) != 1:
                        _status("WARNING: PADDED must be either True or False. "
                                "Only the first specified value was "
                                "considered.", args)
                    timing[argname] = timing[argname][0]
                else:
                    _status("WARNING: {} is not a valid timing parameter, "
                            "therefore it was discarded.".format(argname), args)
        else:
            # Verify that all timing parameters have been set
            if any([val is None for val in timing.values()]):
                msg = "ERROR: Not all timing parameters were set."
                _status(msg, args)
                raise ValueError(msg)
            else:
                _status("SUCCESS: Timing descriptor file was successfully read "
                        "from '{}'.".format(timing_file), args)
                return timing


def _parse_bio_file(bio_file, args, column_order=BIOFILE_COLUMN_ORDER):
    """Reads the data from the physiological data file. Returns the data as 2D
    numpy array, in which columns follow the order (from left to right):
    trigger, cardiac, respiratory. The rearrangement of the columns depends on
    the built-in constant BIOFILE_COLUMN_ORDER."""

    # Update status
    _status("Parsing physiological data file...", args)

    if not os.path.isfile(bio_file):
        msg = "The provided physiological data file at '{}' does not exist."
        _status(msg.format(bio_file), args)
        raise NothingDoneException(msg)
    else:
        try:
            biodata = np.loadtxt(bio_file, dtype=np.float64)
        except:
            msg = "The physiological data file could not be read from '{}' " \
                  "or it contains non-numeric values.".format(bio_file)
            _status(msg, args)
            raise GenericIOException(msg)

        # Rearrange valuable columns, so that it is always in the following
        # order from left to right: trigger, cardiac, respiratory.
        # (This is a built-in convention.)
        biodata = \
            biodata[:, np.array((column_order['trigger'],
                                column_order['cardiac'],
                                column_order['respiratory']))]

        # Update status and return re-arranged dataset
        _status("SUCCESS: The physiological data file was successfully read "
                "from '{}'.".format(bio_file), args)
        return biodata


def _find_peaks(signal, minsep=None):
    """Finds positive peaks in a 1-D signal using the peakutils library. The
    minimum separation of peaks is guessed from the dominant component of the
    demeaned Fourier-transformed signal, but it can also be set explicitly."""

    # Demean signal
    _signal = signal - np.mean(signal)

    if minsep is None:
        # Run FFT on the signal
        # Note that the frequency scale has been normalised so that it does not
        # depend on the exact value of the biopac's sampling interval.
        fft_signal = np.abs(np.fft.rfft(_signal))
        fft_freqs = np.fft.rfftfreq(_signal.size, 1)    # here

        # Find dominant Fourier component
        fft_max_freq = fft_freqs[np.argmax(fft_signal)]
        minsep = int(round(1.0 / fft_max_freq * 2/3.0))      # and here
    else:
        try:
            minsep = int(round(minsep))
        except:
            raise ValueError("The minsep argument must be an integer.")

    # Find peaks
    peak_indices = peakutils.indexes(_signal, thres=0.02 / max(_signal),
                                     min_dist=minsep)

    return np.array(peak_indices)


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
            if not os.path.isdir(dirname):
                if not args['auto']:
                    if confirmed_to_proceed("CONFIRM: Creating '{}': "
                       .format(os.path.join(args['id'], dirname)),
                                            forceanswer=False):
                        _mkdir_p(dirname)
                    else:
                        _status("WARNING: '{}' was not created."
                                .format(os.path.join(args['id'], dirname)),
                                args)
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

    # FIXME: Copying is a very sensitive thing here. This must be reviewed.

    # Update status
    _status("Calculating field map...", args)

    # Specify target directory
    try:
        targetdir = _secure_path(os.path.join(args['id'], FMAP_DIR), args)
    except:
        targetdir = args['id']

    if not args['copy']:
        # The brain-extracted magnitude image will have to be next to the
        # magnitude image.
        targetdir = os.path.split(args['fmag'])[0]
    else:
        # Copy and rename the magnitude image.
        new_fmag = os.path.join(targetdir, args['label'] + FMAG_TAG)
        try:
            _copy(args['fmag'], new_fmag, args, key="fmag",
                  description="field map magnitude image")
        except:
            # TODO: Add exception handling
            # Try to do as much as possible, so pass.
            # FIXME: This error might remain hidden if the second copy is
            # successful.
            pass

        # Copy and rename the phase difference image.
        new_fphase = os.path.join(targetdir, args['label'] + FPHASE_TAG)
        try:
            _copy(args['fphase'], new_fphase, args, key="fphase",
                  description="field map phase difference image")
        except:
            # FIXME: Add exception handling
            raise

    # Run brain extraction on magnitude image (by calling bet from FSL)
    # Please note that the "_brain.nii.gz" is not hard coding but a necessity
    # for FEAT (see hint for B0 unwarping in Feat_gui).
    bet_mag_name = os.path.split(args['fmag'])[1]
    bet_mag_name = str(bet_mag_name).replace(".nii.gz", "_brain.nii.gz")
    args['fmag_brain'] = os.path.join(targetdir, bet_mag_name)
    betcmd = [os.path.join(args['fsldir'], "bin/bet"), args['fmag'],
              args['fmag_brain'], '-f', str(args['fractint'])]
    try:
        _run(betcmd, args, bg=False)
    except NothingDoneException:
        # FIXME: Add exception handling
        raise

    # Prepare the field map (by calling fsl_prepare_fieldmap)

    fpfcmd = [os.path.join(args['fsldir'], "bin/fsl_prepare_fieldmap"),
              'SIEMENS', args['fphase'],
              args['fmag_brain'],
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

            # Copy the file
            new_fmap = os.path.join(targetdir, args['label'] + FMAP_TAG)
            try:
                _copy(args['fmap'], new_fmap, args, key="fmap",
                      description="field map")
            except:
                # FIXME: Add exception handling
                raise


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

        # Copy the file
        new_struct = os.path.join(targetdir, args['label'] + ANAT_TAG)
        try:
            _copy(args['struct'], new_struct, args, key="struct",
                  description="structural image")
        except:
            # TODO: Add exception handling
            # Try to do as much as possible, so ignore the failure for now.
            raise


def copy_single_echo_to_bids(args):
    """Copy single-echo functional scan, the single-echo reference image, and
    the acquisition timing descriptor file into the respective BIDS
    sub-directory (/func)."""

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
        new_secho = os.path.join(targetdir, args['label'] + SECHO_TAG)
        try:
            _copy(args['single_echo'], new_secho, args, key='single_echo',
                  description="single-echo functional image")
        except:
            # TODO: Add exception handling
            # FIXME: This error might remain hidden, if the second copy is
            # successful.
            # Try to do as much as possible, so ignore the failure for now.
            pass

        # Copy the single-echo reference image
        new_sref = os.path.join(targetdir, args['label'] + SREF_TAG)
        try:
            _copy(args['sref'], new_sref, args, key='sref',
                  description="single-echo reference image")
        except:
            # TODO: Add exception handling
            # Try to do as much as possible, so ignore the failure for now.
            pass

        # Copy the acquisition timing descriptor file
        new_stime = os.path.join(targetdir, args['label'] + STIME_TAG)
        try:
            _copy(args['stime'], new_stime, args, key='stime',
                  description="single-echo timing descriptor file")
        except:
            # TODO: Add exception handling
            # FIXME: This error might remain hidden, if the second copy is
            # successful.
            # Try to do as much as possible, so ignore the failure for now.
            raise


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
        new_mecho = os.path.join(targetdir, args['label'] + MECHO_TAG)
        try:
            _copy(args['multi_echo'], new_mecho, args, key='multi_echo',
                  description="multi-echo functional image")
        except:
            # TODO: Add exception handling
            # Try to do as much as possible, so ignore the failure for now.
            pass

        # Copy the multi-echo reference image
        new_mref = os.path.join(targetdir, args['label'] + MREF_TAG)
        try:
            _copy(args['mref'], new_mref, args, key='mref',
                  description="multi-echo reference image")
        except:
            # TODO: Add exception handling
            # Try to do as much as possible, so ignore the failure for now.
            pass

        # Copy the acquisition timing descriptor file
        new_mtime = os.path.join(targetdir, args['label'] + MTIME_TAG)
        try:
            _copy(args['mtime'], new_mtime, args, key='mtime',
                  description="multi-echo timing descriptor file")
        except:
            # TODO: Add exception handling
            # FIXME: This error might remain hidden, if the second copy is
            # successful.
            # Try to do as much as possible, so ignore the failure for now.
            raise


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
            args['sechev'] = fname
        except:
            msg = "ERROR: The cheating EV file could not be saved to '{}'."\
                  .format(fname)
            _status(msg, args)
            raise GenericIOException(msg)

    # MULTI-ECHO (Not used at the moment)
    # FIXME: Need to test if this is really required.
    """
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
            args['mechev'] = fname
        except:
            msg = "ERROR: The cheating EV file could not be created at '{}'."\
                  .format(fname)
            _status(msg, args)
            raise GenericIOException(msg)
    """


def pad_single_echo(args):
    """Adds a number (N_PAD_SLICES) of zero-slices to both ends of the
    single-echo images along the z axis. This is for compatibility with the
    motion correction algorithm in the FSL."""

    # Single-echo functional image
    _status("Padding the single-echo functional image with {} zero-slices on "
            "both ends along the z axis...".format(N_PAD_SLICES), args)
    if not args['single_echo']:
        msg = "ERROR: Single-echo functional image was not specified."
        _status(msg, args)
        raise NotFoundException(msg)
    else:
        # Define padded image path and name
        # FIXME: Hard coding the extension is bad practice.
        fpath, fname = os.path.split(args['single_echo'])
        fname = fname.replace(".nii.gz", "_padded.nii.gz")
        fname = os.path.join(fpath, fname)

        # Check whether the defined file already exist
        do_padding = True
        if os.path.isfile(fname):
            if not args['auto']:
                if confirmed_to_proceed("Would you like to use the existing "
                                        "padded single-echo functional image? "
                                        "(y/n): "):
                    do_padding = False
        if do_padding:
            padded_nifti = _pad(args['single_echo'], n_slices=N_PAD_SLICES)
            # Save the padded single-echo functional image
            try:
                nib.save(padded_nifti, fname)
                _status("Padded single-echo functional image was saved to '{}'"
                        .format(fname), args)
            except:
                # Suppress error and try to do as many tasks as possible.
                msg = "ERROR while saving the padded single-echo functional " \
                      "image to '{}'.".format(fname)
                _status(msg, args)
                pass

        # Update information in program argument dictionary
        args['single_echo_pad'] = fname
        _status("Path to the single-echo functional image was set to "
                "the padded image: '{}'".format(fname), args)

    # Single-echo reference image
    _status("Padding the single-echo reference image with {} zero-slices "
            "on both ends along the z axis...".format(N_PAD_SLICES), args)
    if not args['sref']:
        msg = "ERROR: Single-echo reference image was not specified."
        _status(msg, args)
        raise NotFoundException(msg)
    else:
        # Define padded image path and name
        # FIXME: Hard coding the extension is bad practice.
        fpath, fname = os.path.split(args['sref'])
        fname = fname.replace(".nii.gz", "_padded.nii.gz")
        fname = os.path.join(fpath, fname)

        # Check whether the defined file already exist
        do_padding = True
        if os.path.isfile(fname):
            if not args['auto']:
                if confirmed_to_proceed("Would you like to use the existing "
                                        "padded single-echo reference image? "
                                        "(y/n): "):
                    do_padding = False
        if do_padding:
            padded_nifti = _pad(args['sref'], n_slices=N_PAD_SLICES)
            # Save the padded single-echo reference image
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

        # Update information in program argument dictionary
        args['sref_pad'] = fname
        _status("Path to the single-echo reference image was set to "
                "the padded image: '{}'".format(fname), args)


def pad_multi_echo(args):
    """Adds a number (N_PAD_SLICES) of zero-slices to both ends of the
    multi-echo images along the z axis. This is for compatibility with the
    motion correction algorithm in the FSL."""

    # Multi-echo functional image
    _status("Padding the multi-echo functional image with {} zero-slices on "
            "both ends along the z axis...".format(N_PAD_SLICES), args)
    if not args['multi_echo']:
        msg = "ERROR: Multi-echo functional image was not specified."
        _status(msg, args)
        raise NotFoundException(msg)
    else:
        # Define padded image path and name
        # FIXME: Hard coding the extension is bad practice.
        fpath, fname = os.path.split(args['multi_echo'])
        fname = fname.replace(".nii.gz", "_padded.nii.gz")
        fname = os.path.join(fpath, fname)

        # Check whether the defined file already exist
        do_padding = True
        if os.path.isfile(fname):
            if not args['auto']:
                if confirmed_to_proceed("Would you like to use the existing "
                                        "padded multi-echo functional image? "
                                        "(y/n): "):
                    do_padding = False

        if do_padding:
            padded_nifti = _pad(args['multi_echo'], n_slices=N_PAD_SLICES)
            # Save the padded multi-echo functional image
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

        # Update information in program argument dictionary
        args['multi_echo_pad'] = fname
        _status("Path to the multi-echo functional image was set to "
                "the padded image: '{}'".format(fname), args)

    # Single-echo reference image
    _status("Padding the multi-echo reference image with {} zero-slices "
            "on both ends along the z axis...".format(N_PAD_SLICES), args)
    if not args['mref']:
        msg = "ERROR: Multi-echo reference image was not specified."
        _status(msg, args)
        raise NotFoundException(msg)
    else:
        # Define padded image path and name
        # FIXME: Hard coding the extension is bad practice.
        fpath, fname = os.path.split(args['mref'])
        fname = fname.replace(".nii.gz", "_padded.nii.gz")
        fname = os.path.join(fpath, fname)

        # Check whether the defined file already exist
        do_padding = True
        if os.path.isfile(fname):
            if not args['auto']:
                if confirmed_to_proceed("Would you like to use the existing "
                                        "padded multi-echo reference image? "
                                        "(y/n): "):
                    do_padding = False

        if do_padding:
            padded_nifti = _pad(args['mref'], n_slices=N_PAD_SLICES)
            # Save the padded multi-echo reference image
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

        # Update information in program argument dictionary
        args['mref_pad'] = fname
        _status("Path to the multi-echo reference image was set to the "
                "padded image: '{}'".format(fname), args)


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
            new_anat = os.path.join(args['id'], STRUCT_BASE_NAME + ".anat")
            if os.path.isdir(new_anat):
                args['anatdir'] = new_anat
                _status("Path to the .anat directory was set to {}"
                        .format(new_anat), args)
            else:
                _status("ERROR: Path to the .anat directory could not be set "
                        "after running fsl_anat.", args)
        except NothingDoneException:
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
    """This sub-routine runs FEAT with the required settings, using the input
    from the program arguments."""

    # Update status
    _status("Running FEAT... This might take as long as 3-5 hours.", args)

    # Create copy of template design file and edit it accordingly
    fsf_template_path = os.path.join(args['progdir'], "templates/feat.fsf")
    current_fsf_path = os.path.join(args['id'], "feat_design.fsf")
    if not os.path.isfile(fsf_template_path):
        raise ImportError("FEAT configuration file (.fsf) template could not "
                          "be loaded from {}".format(fsf_template_path))
    else:
        shutil.copy2(fsf_template_path, current_fsf_path)

    # Open timing file
    try:
        timing = _parse_timing_file(args['stime'], args)
        TR = timing['TR']
        TE = timing['TE'][0]
    except:
        msg = "The single-echo acquisition timing descriptor file could not " \
              "be loaded from '{}'.".format(args['stime'])
        _status(msg, args)
        raise NotFoundException(msg)

    # Load shape information from the single-echo functional image
    try:
        single_echo_shape = nib.load(args['single_echo_pad'])\
                            .header.get_data_shape()
    except:
        msg = "The single-echo functional image could not be loaded from '{}'."\
              .format(args['stime'])
        _status(msg, args)
        raise NotFoundException(msg)

    if len(single_echo_shape) < 4:
        msg = "The single-echo functional image had invalid shape: {}"\
            .format(single_echo_shape)
        _status(msg, args)
        raise NIFTIException(msg)

    # Load bias-corrected brain-extracted image from the .anat directory
    struct_biascorr_brain = os.path.realpath(glob.glob(
        os.path.join(args['anatdir'], "*_biascorr_brain.nii.gz"))[0])

    # Gather all information
    fsfdata = \
        {"$OUTPUTDIR": "\"" + os.path.join(args['id'], FEAT_DIR) + "\"",
         "$TR": TR/1000.0,
         "$VOLUMES": single_echo_shape[3],
         "$DT": args['echodiff'],
         "$TE": TE,
         "$SMOOTH": 0.0,
         "$STANDARD":
             "\"" + os.path.join(args['fsldir'],
                                 "data/standard/MNI152_T1_2mm_brain") + "\"",
         "$VOXELS": np.prod(single_echo_shape),
         "$FUNC": "\"" + args['single_echo_pad'] + "\"",
         "$REF": "\"" + args['sref_pad'] + "\"",
         "$FMAP": "\"" + args['fmap'] + "\"",
         "$MAG_FMAP_BRAIN": "\"" + args['fmag_brain'] + "\"",
         "$STRUCT": "\"" + struct_biascorr_brain + "\"",
         "$EV": "\"" + args['sechev'] + "\""}

    # Create a working instance of a FEAT configuration file
    with open(current_fsf_path, mode="r") as design_file:
        filedata = design_file.read()
    for key in fsfdata.keys():
        filedata = filedata.replace(key, str(fsfdata[key]))
    with open(current_fsf_path, mode="w") as design_file:
        design_file.write(filedata)

    # Run FEAT in the background
    featcmd = [os.path.join(args['fsldir'], "bin/feat"),
               os.path.realpath(current_fsf_path)]
    _run(featcmd, args, bg=True)

    # Update the feat directory in the program argument dictionary
    featdir = os.path.join(args['id'], FEAT_DIR + ".feat")
    if os.path.isdir(featdir):
        args['sfeat'] = featdir
    else:
        msg = "Path to the FEAT output directory could not be set to '{}' " \
              "after running FEAT. Please re-run the program and load the " \
              "FEAT directory manually.".format(featdir)
        _status(msg, args)
        raise GenericIOException(msg)


def load_featdir(args):
    """Loads existing output directory from a previous FEAT session."""

    if not os.path.isdir(args['sfeat']):
        _status("ERROR: The provided FEAT directory does not exist."
                .format(args['sfeat']), args)
    else:
        _status("SUCCESS: Path to FEAT directory was set to '{}'."
                .format(args['sfeat']), args)


def load_biodata(args):
    """Loads and copies (upon request) the available physiological data files.
    """

    # Update status
    _status("Loading physiological data...", args)

    # Load biodata for the single-echo scan
    if args['sbio']:
        if not os.path.isfile(args['sbio']):
            _status("The provided physiological data file at '{}' does not "
                    "exist.".format(args['sbio']), args)
        else:
            if not args['copy']:
                _status("Path to the physiological data file (single-echo "
                        "scan) was set to '{}'".format(args['sbio']), args)
            else:
                targetdir = os.path.join(args['id'], BIO_DIR)
                if not os.path.isdir(targetdir):
                    _mkdir_p(targetdir)
                new_sbio = os.path.join(targetdir, args['label'] + SBIO_TAG)

                try:
                    _copy(args['sbio'], new_sbio, args, key="sbio",
                          description="physiological data file "
                                      "(single-echo scan)")
                except:
                    # FIXME: Add exception handling
                    pass

    # Load biodata for the multi-echo scan
    if args['mbio']:
        if not os.path.isfile(args['mbio']):
            _status("The provided physiological data file at '{}' does not "
                    "exist.".format(args['mbio']), args)
        else:
            if not args['copy']:
                _status("Path to the physiological data file (multi-echo "
                        "scan) was set to '{}'".format(args['mbio']), args)
            else:
                targetdir = os.path.join(args['id'], BIO_DIR)
                if not os.path.isdir(targetdir):
                    _mkdir_p(targetdir)
                new_mbio = os.path.join(targetdir, args['label'] + MBIO_TAG)

                try:
                    _copy(args['mbio'], new_mbio, args, key="mbio",
                          description="physiological data file "
                                      "(multi-echo scan)")
                except:
                    # FIXME: Add exception handling
                    raise


def single_echo_analysis(args):
    """The implementation of the analysis steps pertaining to the single-echo
    data."""

    # Update status
    _status("Starting single-echo analysis...", args)

    # Create BIDS sub-directory for masks
    maskdir = os.path.join(args['id'], MASK_DIR)
    try:
        _mkdir_p(maskdir)
        _status("BIDS sub-directory for masks was successfully created at "
                "'{}'.".format(maskdir), args)
        # Update the information of mask directory in the program argument
        # dictionary
        args['maskdir'] = maskdir
    except:
        _status("WARNING: BIDS sub-directory for masks could not be created. "
                "Masks will be saved into the subject directory.", args)
        args['maskdir'] = args['id']

    # Erode bias-corrected brain-extracted structural image (img) and save into
    # the BIDS directory for masks.
    # (calls fslmaths)
    source_img = glob.glob(os.path.join(args['anatdir'],
                                        "*_biascorr_brain.nii.gz"))[0]
    struct_base_name = os.path.split(source_img)[-1]\
        .replace("_biascorr_brain.nii.gz", "")
    if not os.path.isfile(source_img):
        msg = "ERROR: The bias-corrected brain-extracted structural image "\
              "was not found in the .anat directory ('{}')."\
              .format(source_img)
        _status(msg, args)
        raise NotFoundException(msg)
    else:
        ero_img = os.path.join(args['maskdir'], ERO_NAME)
        try:
            _copy(source_img, ero_img, args,
                  description="bias-corrected brain-extracted structural image")
        except:
            raise
        erocmd = [os.path.join(args['fsldir'], "bin/fslmaths"), ero_img,
                  "-ero", ero_img]
        try:
            for _ in range(N_EROSIONS):
                _run(erocmd, args, bg=False)
        except:
            raise

    # Use FEAT output to warp structural images into functional space
    # (calls fsl invwarp and applywarp)

    # Calculate inverse warp
    func = os.path.join(args['sfeat'], "reg/example_func.nii.gz")
    func2st_warp = os.path.join(
        args['sfeat'], "reg/example_func2highres_warp.nii.gz")
    st2func_warp = os.path.join(
        args['sfeat'], "reg/highres2example_func.nii.gz")
    invwarpcmd = [os.path.join(args['fsldir'], "bin/invwarp"), "-w",
                  func2st_warp, "-o", st2func_warp, "-r", func]
    try:
        _run(invwarpcmd, args, bg=False)
    except:
        raise

    # Apply the inverted warp to the structural images in the .anat directory
    # TODO: This could be run in parallel
    _status("Warping structural images into functional space...", args)

    for (k, v) in {struct_base_name + "_biascorr_brain.nii.gz":
                   struct_base_name, struct_base_name + "_fast_pve_0.nii.gz":
                   "CSF", struct_base_name + "_fast_pve_1.nii.gz": "GM",
                   struct_base_name + "_fast_pve_2.nii.gz": "WM", ero_img:
                   ero_img.replace(".nii.gz", "")}.iteritems():

        appwcmd = [os.path.join(args['fsldir'], "bin/applywarp"),
                   "-i", os.path.join(args['anatdir'], k),
                   "-o", os.path.join(args['maskdir'], v + "2func.nii.gz"),
                   "-r", func,
                   "-w", st2func_warp]
        try:
            _run(appwcmd, args, bg=True)
        except:
            continue

    # Load functional data: the residuals from the FEAT session
    _status("Loading residuals from FEAT session...", args)
    res_path = os.path.join(args['sfeat'], "stats/res4d.nii.gz")
    if not os.path.isfile(res_path):
        msg = "The 4D NIfTI image of residuals was not found at '{}'."\
            .format(res_path)
        _status(msg, args)
        raise NotFoundException(msg)
    else:
        try:
            residuals = nib.load(res_path)
            hdr = residuals.header
            residuals = np.array(residuals.get_data())
            residuals_shape = residuals.shape
            _status("SUCCESS: 4D NIfTI image of residuals was successfully "
                    "loaded from '{}'.".format(res_path), args)
        except:
            msg = "ERROR: 4D NIfTI image of residuals could not be loaded " \
                  "from '{}'.".format(res_path)
            _status(msg, args)
            raise NIFTIException(msg)

    # Read acquisition timing information
    try:
        timing = _parse_timing_file(args['stime'], args)
        # Introduce variables for better readability
        TR = timing['TR']
        SS = int(round(timing['SS']))
        # TE = timing['TE']
        if timing['PADDED']:
            ST = timing['ST']
        else:
            ST = timing['ST']
            ST = np.concatenate((np.repeat(ST[0], N_PAD_SLICES), ST,
                                 np.repeat(ST[-1], N_PAD_SLICES)))
    except:
        # TODO: Add exception handling
        raise

    # Import physiological data
    try:
        biodata = _parse_bio_file(args['sbio'], args)
    except:
        # TODO: Add exception handling
        raise

    # Trim physiological data (to match scan duration) before the iterative GLM
    # fitting. Discard first few (timing['SS']) acquisitions until steady state.

    samples_per_ms = args['sfreq'] / 1000.0
    samples_per_TR = int(round(samples_per_ms * timing['TR']))
    trigger_threshold = \
        np.mean(KMeans(n_clusters=2).fit(biodata[:, 0].reshape(-1, 1))
                .cluster_centers_)
    trigger_duration = TRIGGER_DURATION * samples_per_ms
    trigger_on = np.where(biodata[:, 0] > trigger_threshold)[0]
    ss_begins = int(ceil(np.min(trigger_on) +
                         timing['SS'] * timing['TR'] * samples_per_ms))
    scan_start = int(np.min(trigger_on[trigger_on > ss_begins]))
    scan_end = int(np.max(trigger_on[trigger_on > ss_begins]) + samples_per_TR
                   - trigger_duration)

    # Sub-sample physiological data (to match TR) before the iterative GLM
    # fitting.

    cardiac_signal = biodata[scan_start:scan_end+1:samples_per_TR, 1]\
        .reshape((-1, 1))
    respiratory_signal = biodata[scan_start:scan_end+1:samples_per_TR, 2]\
        .reshape((-1, 1))

    # Run iterative GLM fitting
    _status("Starting iterative GLM fitting...", args)
    ev = np.hstack((respiratory_signal, cardiac_signal))
    voxel_signals = residuals.reshape((-1, residuals_shape[-1]))
    voxel_signals = voxel_signals[:, SS:]
    print "Design matrix:", ev.shape
    print "Signals from all voxels:", voxel_signals.shape
    try:
        glm = GLMObject(ev, voxel_signals, 1000.0/TR, args['sfmin'],
                        args['spval'], args['sconv'])\
            .fit(n_jobs_=-1, total_n_EVs=3, iterations=True, normalize=True,
                 verbose=args['verbose'])
        fft_voxels, fft_EVs, fft_freq_range, coef_initial, coef_refined = glm
        _status("SUCCESS: The iterative GLM fitting was successfully "
                "completed.", args)
    except:
        _status("ERROR in the iterative GLM fitting.", args)
        # TODO: Add exception handling
        raise

    # Obtain respiratory and cardiac maps
    coef_initial = \
        coef_initial.reshape(residuals_shape[:3] + (fft_EVs.shape[-1],))
    respiratory_map = coef_initial[:, :, :, 1]
    respiratory_map[respiratory_map < 0] = 0
    cardiac_map = coef_initial[:, :, :, 2]
    cardiac_map[cardiac_map < 0] = 0

    # Save both the respiratory map and the cardiac map
    targetdir = os.path.join(args['id'], RESULTS_DIR)
    if not os.path.isdir(targetdir):
        _mkdir_p(targetdir)
    args['resultsdir'] = targetdir
    brain_mask = \
        os.path.join(args['maskdir'], struct_base_name + "2func.nii.gz")
    hdr = nib.load(brain_mask).header

    cardmap_path = \
        os.path.join(args['resultsdir'], args['label'] + CARDMAP_TAG)
    try:
        nib.save(nib.Nifti1Image(cardiac_map, hdr.get_sform(), hdr),
                 cardmap_path)
        _status("SUCCESS: The cardiac map was successfully saved to '{}'."
                .format(cardmap_path), args)
    except:
        _status("ERROR: The cardiac map could not be saved to '{}'."
                .format(cardmap_path), args)
        # TODO: Add exception handling
        raise

    respmap_path = \
        os.path.join(args['resultsdir'], args['label'] + RESPMAP_TAG)
    try:
        nib.save(nib.Nifti1Image(respiratory_map, hdr.get_sform(), hdr),
                 respmap_path)
        _status("SUCCESS: The respiratory map was successfully saved to '{}'."
                .format(respmap_path), args)
    except:
        _status("ERROR: The respiratory map could not be saved to '{}'."
                .format(respmap_path), args)
        # TODO: Add exception handling
        raise

    # PHASE MAPPING
    # Update status
    _status("Starting phase mapping...", args)

    # Obtain the dominant cardiac frequency
    cardiac_spectrum = fft_EVs[:, 2]
    respiratory_spectrum = fft_EVs[:, 1]
    dom_card_freq_index = np.argmax(cardiac_spectrum[1:])
    dom_card_freq = fft_freq_range[dom_card_freq_index]
    _status("The dominant cardiac frequency is {} Hz ({}/min)."
            .format(dom_card_freq, int(round(dom_card_freq*60))), args)

    # Auto-select reference voxel
    # FIXME: Use a brain map to ensure that the reference voxel is in the brain.
    ref_coords = np.unravel_index(cardiac_map.argmax(), cardiac_map.shape)
    # FIXME: An angiogram would be even more accurate here.
    _status("Coordinates of the arterial reference voxel: {}"
            .format(str(ref_coords)), args)

    # Calculate phase of the dominant cardiac frequency component in the
    # reference voxel
    x, y, z = ref_coords
    ref_phase = np.fft.rfft(residuals[x, y, z, :])[dom_card_freq_index]
    phasediff_multiband = np.zeros(residuals_shape[:-1])
    for slice_no in range(residuals_shape[2]):
        phasediff_multiband[:, :, slice_no] = \
            (ST[slice_no] - ST[z]) / 1000.0 * dom_card_freq * 2*np.pi

    phase_voxels = np.fft.rfft(residuals[:, :, :, SS:], axis=-1)
    freq_index_from_zero = \
        phase_voxels.shape[-1] - fft_voxels.shape[-1] + dom_card_freq_index
    phase_map = \
        np.angle(ref_phase / phase_voxels[:, :, :, freq_index_from_zero]) \
        + phasediff_multiband

    # Express phase in (-pi, pi)
    phase_map[phase_map > np.pi] = -2 * np.pi + phase_map[phase_map > np.pi]
    phase_map[phase_map < -np.pi] = 2 * np.pi + phase_map[phase_map < -np.pi]

    # Remove background
    phase_map[np.logical_not(np.any(phase_voxels, axis=-1))] = 0

    # Save phase map
    try:
        phasemap_path = \
            os.path.join(args['resultsdir'], args['label'] + PHASEMAP_TAG)
        nib.save(nib.Nifti1Image(phase_map, hdr.get_sform(), hdr),
                 phasemap_path)
        _status("SUCCESS: The phase map was successfully saved to '{}'."
                .format(phasemap_path), args)
    except:
        _status("ERROR: The phase map could not be saved to '{}'."
                .format(phasemap_path), args)
        # TODO: Add exception handling
        raise

    # Save interactive charts into the results folder
    # TODO: This feature will be added later.

    # Update status
    _status("SUCCESS: Single-echo analysis was completed successfully.", args)


def prepare_multi_echo(args):
    """Parses the multi-echo functional image to sort the sequence of echos
    with the same index into separate files. Runs motion correction (mcflirt) on
    the echo files."""

    # Update status
    _status("Preparing for multi-echo analysis...", args)

    # Load the multi-echo functional image (the padded)
    if not os.path.isfile(args['multi_echo_pad']):
        msg = "The provided padded multi-echo functional image was not found " \
              "at '{}'.".format(args['multi_echo_pad'])
        _status(msg, args)
        raise NotFoundException(msg)
    else:
        try:
            mimg = nib.load(args['multi_echo_pad'])
            hdr = mimg.header
            args['mhdr'] = copy.deepcopy(hdr)
            mimg = mimg.get_data()
        except:
            msg = "The padded multi-echo functional image could not be " \
                  "loaded from '{}'.".format(args['multi_echo_pad'])
            _status(msg, args)
            raise GenericIOException(msg)

    # Load the timing parameters for the multi-echo acquisition
    try:
        timing = _parse_timing_file(args['mtime'], args)
        # Introduce variables for better readability
        TR = timing['TR']
        SS = int(round(timing['SS']))
        TE = timing['TE']
        if timing['PADDED']:
            ST = timing['ST']
        else:
            ST = timing['ST']
            ST = np.concatenate((np.repeat(ST[0], N_PAD_SLICES), ST,
                                 np.repeat(ST[-1], N_PAD_SLICES)))
    except:
        # TODO: Add exception handling
        raise

    # SORTING ECHOS INTO SEPARATE TIME SERIES (4D NIfTI IMAGES)
    n_echos = TE.size
    if not (mimg.shape[-1] % n_echos == 0):
        _status("WARNING: The number of volumes in the multi-echo functional "
                "image is not a multiple of the number of echos, as specified "
                "in the acquisition timing descriptor file.", args)
    targetdir = os.path.join(args['id'], FUNC_DIR)
    if not os.path.isdir(targetdir):
        _mkdir_p(targetdir)

    # Check whether the separate echo files already exist
    echo_path = os.path.split(args['multi_echo_pad'])[0]
    echo_names = [os.path.join(echo_path, str(args['label'] + MECHO_TAG)
                               .replace(".nii.gz", "_echo{:d}.nii.gz")
                               .format(i)) for i in range(n_echos)]
    do_separation = True
    if all([os.path.isfile(echo_name) for echo_name in echo_names]):
        if not args['auto']:
            if confirmed_to_proceed("Would you like to use the existing set of "
                                    "separate echo images? (y/n): "):
                do_separation = False

    if do_separation:
        for i, echo_name in enumerate(echo_names):
            # Get corresponding echos from consecutive TRs
            current_echo_series = mimg[:, :, :, i::n_echos]
            # Manipulate header to fit the new data
            hdr.set_data_shape(current_echo_series.shape)
            # Save the current echo into file
            try:
                nib.save(nib.Nifti1Image(current_echo_series, hdr.get_sform(),
                                         hdr), echo_name)
                _status("SUCCESS: Echo No. {}. file was successfully saved to "
                        "'{}'".format(i, echo_name), args)
            except:
                _status("ERROR: Echo No. {}. file could not be saved to '{}'"
                        .format(i, echo_name), args)
                # Try to save as much as possible
                continue

    # MOTION CORRECTION (calls mcflirt and applyxfm4D)
    # Find the appropriate series of time-to-time rigid-body transfromations
    # for the first echo (signal is most prominent for this), then apply it to
    # further echos as well. This relies on the assumption that within-TR head
    # motion effect are negligible.
    if not args['fsldir']:
        args['fsldir'] = get_fsldir()

    # Check for existing motion-corrected echo files:
    echo_moco_names = [str(echo).replace(".nii.gz", "_moco.nii.gz")
                       for echo in echo_names]
    run_moco = False
    if all([os.path.isfile(echo) for echo in echo_moco_names]):
        if args['auto'] or (not confirmed_to_proceed(
                "Would you like to use the existing motion-corrected echo "
                "images? (y/n): ", forceanswer=True)):
            run_moco = True

    if run_moco:
        first_echo_moco = echo_names[0].replace(".nii.gz", "_moco")
        mcflirtcmd = [os.path.join(args['fsldir'], "bin/mcflirt"),
                      "-in", echo_names[0],
                      "-out", first_echo_moco,
                      "-reffile", args['mref_pad'],
                      "-mats"]
        try:
            _run(mcflirtcmd, args, bg=False)
        except:
            raise

        for i, echo in enumerate(echo_names[1:]):
            applycmd = [os.path.join(args['fsldir'], "bin/applyxfm4D"),
                        echo,
                        first_echo_moco + ".nii.gz",
                        echo_moco_names[i+1],
                        first_echo_moco + ".mat",
                        "-fourdigit"]
            try:
                _run(applycmd, args, bg=True)
            except NothingDoneException as exc:
                _status(exc.message, args)
                continue

    # Add information about echo files to the program argument dictionary
    args['echo_files'] = echo_moco_names


def multi_echo_analysis(args):
    """Assigns MR signal measurements to cardiac cycle segments. Fits
    exponential curve to the data to export S0 and T2* maps for the entire
    brain."""

    # Update status
    _status("Starting multi-echo analysis...", args)

    # Load timing information
    timing = _parse_timing_file(args['mtime'], args)
    # Introduce variables for better readability
    TR = timing['TR']
    SS = int(round(timing['SS']))
    TE = timing['TE']
    if timing['PADDED']:
        ST = timing['ST']
    else:
        ST = timing['ST']
        ST = np.concatenate((np.repeat(ST[0], N_PAD_SLICES), ST,
                             np.repeat(ST[-1], N_PAD_SLICES)))

    # Load trigger and cardiac signal
    triggers, cardiac_signal = _parse_bio_file(args['mbio'], args)[:, :2].T

    # Update status
    _status("Segmenting cardiac signal...", args)

    # Trim cardiac signal to the duration of steady-state MR acquisition
    # Note that this time the cardiac signal is not sub-sampled.
    samples_per_ms = args['sfreq'] / 1000.0
    samples_per_TR = int(round(samples_per_ms * TR))
    trigger_threshold = np.mean(KMeans(n_clusters=2).fit(
        triggers.reshape((-1, 1))).cluster_centers_)
    trigger_duration = TRIGGER_DURATION * samples_per_ms
    trigger_on = np.where(triggers > trigger_threshold)[0]
    ss_begins = int(ceil(np.min(trigger_on) + SS * TR * samples_per_ms))
    scan_start = int(np.min(trigger_on[trigger_on > ss_begins]))
    scan_end = int(np.max(trigger_on[trigger_on > ss_begins]) + samples_per_TR
                   - trigger_duration)
    cardiac_signal = cardiac_signal[scan_start:scan_end]

    # Find peaks (time points for peak arterial flow) in the cardiac signal
    card_peak_indices = _find_peaks(cardiac_signal)

    # Discard occasional peak artifacts on both ends of the signal
    card_peak_indices = np.setdiff1d(card_peak_indices,
                                     np.array([0, len(cardiac_signal)-1]))

    # Create a 1-D segment mask for the cardiac signal so that each period
    # consists of a pre-specified number of segments.
    card_segment_mask = np.zeros_like(cardiac_signal)
    binwidth = []

    # For full cycles
    for peak_no in range(len(card_peak_indices) - 1):
        cycle_start = card_peak_indices[peak_no]
        cycle_end = card_peak_indices[peak_no + 1]
        period = int(cycle_end - cycle_start)
        binwidth.append(int(ceil(period / float(args['cseg']))))
        # Segment labels must be different from 0.
        segments = np.concatenate([[i] * binwidth[-1]
                                   for i in range(1, args['cseg']+1)])
        card_segment_mask[cycle_start:cycle_end] = segments[:period]

    # For the potentially incomplete leading and trailing cycles
    binwidth = np.mean(binwidth).astype(np.int64)
    # Segment labels must be different from 0.
    segments = np.concatenate([[i] * binwidth
                               for i in range(1, args['cseg']+1)])
    cycle_end = card_peak_indices[0]
    cycle_start = card_peak_indices[-1]
    # Leading end of cardiac signal
    card_segment_mask[cycle_end::-1] = \
        segments[:args['cseg'] * binwidth - cycle_end - 2:-1]
    # Trailing end of cardiac signal
    card_segment_mask[cycle_start:] = \
        segments[:card_segment_mask.size - cycle_start]

    # Load all echos
    _status("Loading echo time series...", args)
    echos_exist = [os.path.isfile(echo) for echo in args['echo_files']]
    if not all(echos_exist):
        msg = "The following echo file(s) could not be found: {}"\
              .format("\n".join(args['echo_files'][echos_exist is False]))
        _status(msg, args)
        raise NotFoundException(msg)
    try:
        # TODO: Add support for non-equal length files (very rare)
        all_echos = np.concatenate(
            [nib.load(echo).get_data()[:, :, :, :, np.newaxis]
             for echo in args['echo_files']], axis=-1)
    except:
        msg = "Not all echo files had the same number of time points."
        _status(msg, args)
        raise NIFTIException(msg)

    # Calculate S0 and T2* for every TR in every voxel

    _status("Calculating S0 and T2* for every acquisition in each voxel...",
            args)
    echos_shape = all_echos.shape
    log_echos = np.log(all_echos.reshape((-1, echos_shape[-1])))
    params = np.zeros(log_echos.shape[:-1] + (2,))
    truevals = \
        np.where(
            np.logical_not(np.logical_or(np.any(np.isinf(log_echos), axis=-1),
                                         np.any(np.isnan(log_echos), axis=-1),
                                         np.any(np.isneginf(log_echos),
                                                axis=-1))))[0]
    log_echos = log_echos[truevals, :]
    params[truevals, :] = np.polyfit(TE, log_echos.T, 1).T
    params = params.reshape(echos_shape[:-1] + (2,))
    T2star = -1.0 / params[:, :, :, :, 0]
    S0 = np.exp(params[:, :, :, :, 1])

    # Bin the repeated measurements of S0 and T2* into the segments of the
    # cardiac cycle (when each of them was measured).
    _status("Binning echo signals, S0, and T2* values into cardiac cycle "
            "segments...", args)

    # Create an array that stores the echo signals' coincidence with cardiac
    # cycle segments.
    all_echos_segmented = \
        np.zeros(echos_shape[:3] + (echos_shape[3],) + (echos_shape[-1],))
    all_echotrains_segmented = np.zeros(echos_shape[:3] + (echos_shape[3],))

    for repeat_no in range(SS, echos_shape[-2]):
        # Calculate the segments for individual echos
        for echo_no in range(echos_shape[-1]):
            index_base = (TE[echo_no] + (repeat_no-SS) * TR) \
                         * samples_per_ms
            for slice_no in range(echos_shape[2]):
                index = int(round(index_base + ST[slice_no]))
                all_echos_segmented[:, :, slice_no, repeat_no, echo_no] \
                    = card_segment_mask[index]

        # Calculate the segments for echo trains
        index_base = \
            (TE[0] + np.mean(TE) + (repeat_no-SS) * TR) * samples_per_ms
        for slice_no in range(echos_shape[2]):
            index = int(round(index_base + ST[slice_no]))
            all_echotrains_segmented[:, :, slice_no, repeat_no] = \
                card_segment_mask[index]

    # Check whether enough data is available.

    # Set output directory
    targetdir = os.path.join(args['id'], RESULTS_DIR)
    if not os.path.isdir(targetdir):
        _mkdir_p(targetdir)

    # Set output shape and adjust NIfTI header
    stat_shape = echos_shape[:3] + (args['cseg'],)
    hdr_stat = copy.deepcopy(args['mhdr'])
    hdr_stat.set_data_shape(stat_shape)

    # Calculate mean signal intensity and standard deviation for each cardiac
    # cycle segment.
    for echo_no in range(echos_shape[-1]):
        # Initialise containers
        mean_signal_by_segment = np.zeros(stat_shape)
        stderr_signal_by_segment = np.zeros(stat_shape)

        # Calculate the statistics
        for segment_no in range(args['cseg']):
            coords = np.where(all_echos_segmented[:, :, :, :, echo_no] ==
                              segment_no + 1)
            tmp = np.full(echos_shape[:-1], np.nan)
            tmp[coords] = all_echos[coords + (echo_no,)]
            mean_signal_by_segment[coords[:3] + (segment_no,)] \
                = np.nanmean(tmp, axis=-1)[coords[:3]]
            tmp2 = \
                np.nanstd(tmp, axis=-1) / \
                np.sqrt(np.count_nonzero(~np.isnan(tmp), axis=-1))
            del tmp
            stderr_signal_by_segment[coords[:3] + (segment_no,)] = \
                tmp2[coords[:3]]
            del tmp2

        # Save the MEAN map
        try:
            fname = os.path.join(targetdir, args['label'] +
                                 "_echo{:d}_segment_mean.nii.gz"
                                 .format(echo_no))
            nib.save(nib.Nifti1Image(mean_signal_by_segment,
                                     hdr_stat.get_sform(), hdr_stat), fname)
            _status("SUCCESS: An image of cardiac segment-specific mean "
                    "signals for Echo No. {}. was successfully saved to '{}'."
                    .format(echo_no, fname), args)
        except:
            _status("ERROR: An image of cardiac segment-specific mean signals "
                    "for Echo No. {}. could not be saved to '{}'."
                    .format(echo_no, fname), args)
            # Try to save as many as possible
            pass

        # Save the STANDARD ERROR map
        try:
            fname = os.path.join(targetdir, args['label'] +
                                 "_echo{:d}_segment_stderr.nii.gz"
                                 .format(echo_no))
            nib.save(nib.Nifti1Image(stderr_signal_by_segment,
                                     hdr_stat.get_sform(), hdr_stat), fname)
            _status("SUCCESS: An image of cardiac segment-specific standard "
                    "errors of the mean signal for Echo No. {}. was "
                    "successfully saved to '{}'.".format(echo_no, fname), args)
        except:
            _status("ERROR: An image of cardiac segment-specific standard "
                    "errors of the mean signal for Echo No. {}. could not be "
                    "saved to '{}'.".format(echo_no, fname), args)
            # Try to save as many as possible
            pass

    # Calculate mean S0 and mean T2* per cardiac cycle segment and calculate
    # their respective standard errors.
    mean_S0_per_segment = np.zeros(stat_shape)
    stderr_S0_per_segment = np.zeros(stat_shape)
    mean_T2star_per_segment = np.zeros(stat_shape)
    stderr_T2star_per_segment = np.zeros(stat_shape)

    for segment_no in range(args['cseg']):
        coords = np.where(all_echotrains_segmented == segment_no + 1)
        tmp = np.full(echos_shape[:-1], np.nan)
        tmp[coords] = S0[coords]
        mean_S0_per_segment[coords[:3] + (segment_no,)] = \
            np.nanmean(tmp, axis=-1)[coords[:3]]
        tmp2 = \
            np.nanstd(tmp, axis=-1) / \
            np.sqrt(np.count_nonzero(~np.isnan(tmp), axis=-1))
        del tmp
        stderr_S0_per_segment[coords[:3] + (segment_no,)] = tmp2[coords[:3]]
        del tmp2
        tmp = np.full(echos_shape[:-1], np.nan)
        tmp[coords] = T2star[coords]
        tmp[tmp < 0] = np.nan
        tmp[np.isinf(tmp)] = np.nan
        mean_T2star_per_segment[coords[:3] + (segment_no,)] = \
            np.nanmean(tmp, axis=-1)[coords[:3]]
        tmp2 = \
            np.nanstd(tmp, axis=-1) / \
            np.sqrt(np.count_nonzero(~np.isnan(tmp), axis=-1))
        del tmp
        stderr_T2star_per_segment[coords[:3] + (segment_no,)] = tmp2[coords[:3]]
        del tmp2

    # Save the segment-specific MEAN S0 map
    try:
        fname = os.path.join(targetdir, args['label'] +
                             "_S0_segment_mean.nii.gz")
        nib.save(nib.Nifti1Image(mean_S0_per_segment, hdr_stat.get_sform(),
                                 hdr_stat), fname)
        _status("SUCCESS: An image of cardiac segment-specific mean S0 values "
                "was successfully saved to '{}'.".format(fname), args)
    except:
        _status("ERROR: An image of cardiac segment-specific mean S0 values "
                "could not be saved to '{}'.".format(fname), args)
        # Try to save as many as possible
        pass

    # Save the STANDARD ERROR OF segment-specific S0 means map
    try:
        fname = os.path.join(targetdir, args['label'] +
                             "_S0_segment_stderr.nii.gz")
        nib.save(nib.Nifti1Image(stderr_S0_per_segment, hdr_stat.get_sform(),
                                 hdr_stat), fname)
        _status("SUCCESS: An image of the standard errors of cardiac "
                "segment-specific mean S0 values was successfully saved to "
                "'{}'.".format(fname), args)
    except:
        _status("ERROR: An image of the standard errors of cardiac "
                "segment-specific mean S0 values could not be saved to '{}'."
                .format(fname), args)
        # Try to save as many as possible
        pass

    # Save the segment-specific MEAN T2* map
    try:
        fname = os.path.join(targetdir, args['label'] +
                             "_T2star_segment_mean.nii.gz")
        nib.save(nib.Nifti1Image(mean_T2star_per_segment, hdr_stat.get_sform(),
                                 hdr_stat), fname)
        _status("SUCCESS: An image of cardiac segment-specific mean T2* values "
                "was successfully saved to '{}'.".format(fname), args)
    except:
        _status("ERROR: An image of cardiac segment-specific mean T2* values "
                "could not be saved to '{}'.".format(fname), args)
        # Try to save as many as possible
        pass

    # Save the STANDARD ERROR OF segment-specific T2* means map
    try:
        fname = os.path.join(targetdir, args['label'] +
                             "_T2star_segment_stderr.nii.gz")
        nib.save(nib.Nifti1Image(stderr_T2star_per_segment,
                                 hdr_stat.get_sform(), hdr_stat), fname)
        _status("SUCCESS: An image of the standard errors of cardiac "
                "segment-specific mean T2* values was successfully saved to "
                "'{}'.".format(fname), args)
    except:
        _status("ERROR: An image of the standard errors of cardiac "
                "segment-specific mean T2* values could not be saved to '{}'."
                .format(fname), args)
        # Try to save as many as possible
        pass


# Program execution starts here
if __name__ == "__main__":
    print usermanual
    exit(0)

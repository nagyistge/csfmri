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

# FIXME: Extend on the introduction and on the rules, as in the notes.
usermanual = \
"""
The CSFMRI (cardio-synchronous fRMI) Analysis Tool provides robust functionality
for the analysis of fMRI signals at the cardiac frequency.

Inputs

   -id      <subjectdir> Path to the subject folder ending with the subject's ID
   -dir     <dirs>       BIDS directories to create inside subject's directory
   -----------------------------------------------------------------------------
   -t1      <structimg>  T1-weighted structural image (3D)
   -anat    <anatdir>    Path to .anat directory (from fsl_anat) (if exists)
   -----------------------------------------------------------------------------
   -se      <seimg>      EPI single-echo functional image (4D)
   -sref    <srefimg>    EPI single-echo reference image (3D)
   -stime   <stimefile>  Single-echo acq. time parameters (TR, TE, skipped TRs,
                         multi-band slice timings)
   -sfeat   <featdir>    Path to the relevant FEAT output directory (if exists)
   -sfmin   <lowfreq>    Lower frequency threshold (Hz) (default: 0.2)
   -spval   <pval>       Significance level (for GLM fit) (default: 0.05)
   -sconv   <convval>    Desired level of convergence (default: 0.1)
   -sbio    <biofile>    BioPac data (Trigger, Resp, Card, Sats in a text file)
   -----------------------------------------------------------------------------
   -me      <meimg>      EPI multi-echo functional image (4D)
   -mref    <mrefimg>    EPI multi-echo reference image (3D)
   -mtime   <mtimefile>  Multi-echo acq. time parameters (TR, TEs, skipped TRs,
                         multi-band slice timings)
   -mbio    <biofile>    BioPac data (Trigger, Resp, Card, Sats in a text file)
   -----------------------------------------------------------------------------
   -fmap    <fmapimg>    Field map (3D) (if exists)
   -fmag    <fmagimg>    Magnitude image for field map calculation (4D)
   -fphase  <fpimg>      Phase difference image for field map calculation (3D)
   -dt      <echodiff>   Echo time difference (ms) for field map calculation
   -fint    <fractint>   Fractional intensity for BET in field map calculation
                         (default: 0.5)
   -----------------------------------------------------------------------------
   -sfreq   <sfreq>      Sampling frequency for BioPac data (default: 1 kHz)
   -cseg    <n_segm>     Number of segments in the cardiac cycle (default: 10)

Options

   -prep                 Prepare: rename the specified input files and move
                         them to their standard location. (use for full proc.)
   -----------------------------------------------------------------------------
   -auto                 Switch off Interactive Mode. (runs w/o confirmations)
   -v                    Verbose Mode (displays progress information)
   -log     <logfile>    Same as Verbose Mode, but saves a log file.
   -config  <configfile> Load input settings from configuration file.
   -cpu     <n_cores>    Optimize tasks for a number of CPU cores. (default: 1)
   
Prerequisite:

    FSL (FMRIB Software Library) 5.0 (https://www.fmrib.ox.ac.uk/fsl)
   
   """

# IMPORTS

import sys
import os
from cl_interface import *
from csfmri_utils import *
import subprocess
from csfmri_tasks import TASK_LIST, TASK_ORDER
import csfmri_tasks
from collections import OrderedDict


# DEFINITIONS AND CODE

# Command-line arguments
CLFLAGS = {'id':            '-id',
           'bids_dirs':     '-dir',
           'struct':        '-t1',
           'anatdir':       '-anat',
           'single_echo':   '-se',
           'sref':          '-sref',
           'stime':         '-stime',
           'sfeat':         '-sfeat',
           'sfmin':         '-sfmin',
           'spval':         '-spval',
           'sconv':         '-sconv',
           'sbio':          '-sbio',
           'multi_echo':    '-me',
           'mref':          '-mref',
           'mtime':         '-mtime',
           'mbio':          '-mbio',
           'fmap':          '-fmap',
           'fmag':          '-fmag',
           'fphase':        '-fphase',
           'echodiff':      '-dt',
           'fractint':      '-fint',
           'sfreq':         '-sfreq',
           'cseg':          '-cseg',
           'copy':          '-copy',
           'auto':          '-auto',
           'verbose':       '-v',
           'log':           '-log',
           'config':        '-config',
           'cpu':           '-cpu'}

# Explicit arguments whose values cannot be set to default without the user
# mentioning the argument.
EXPLICIT_ARGS = {"bids_dirs"}

# Arguments by type
# FIXME: This could be done more elegantly.
STRING_ARGS = {"id", "bids_dirs", "struct", "anatdir", "single_echo", "sref",
                "stime", "sfeat", "sbio", "multi_echo", "mref", "mtime", "mbio",
                "fmap", "fmag", "fphase", "log"}
FLOAT_ARGS = {"sfmin", "spval", "sconv", "echodiff", "fractint", "sfreq"}
INT_ARGS = {"cseg", "cpu"}
BOOL_ARGS = {"copy", "auto", "verbose"}

# Default values for command-line arguments
ARG_DEFAULTS = {
           'id':            os.getcwd(),
           'bids_dirs':     {'anat', 'func', 'fmap', 'masks', 'orig', 'result'},
           'sfmin':         0.2,
           'spval':         0.05,
           'sconv':         0.1,
           'fractint':      0.5,
           'sfreq':         1000,
           'cseg':          10,
           'copy':          False,
           'auto':          False,
           'verbose':       False,
           'cpu':           1}


def config_file_parser(config_file_path):
    """Reads configuration file and translates its content into command-line
    arguments that can be parsed by the sub-routine parse_arguments()."""

    # Initialise argument dictionary that will be filled and returned
    args = dict(zip(CLFLAGS.keys(), [None] * len(CLFLAGS)))

    # Read all lines
    try:
        with open(config_file_path, mode="r") as f:
            lines = f.readlines()
    except:
        raise IOError("ERROR: Configuration file could not be opened from "
                      "'{}'.".format(config_file_path))

    # Remove blank lines
    lines = [line for line in lines if line]

    # Concatenate lines that end with a line continuation character (\)
    # Note that the \ character is considered as a line continuation only at the
    # end of each line if and only if there is whitespace on both sides of it.
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
            # If no whitespace on the left, discard and move to the next line.
            else:
                break
        i += 1
    # Discard accidental line continuation in the last line
    lines[-1] = lines[-1].replace("\\\n", "\n")

    # Discard comment lines and lines without assignment
    lines = [line for line in lines
             if (not line.startswith("#")) and (line.find("=") != -1)]


    # Create argument dictionary from argument flags without the leading '-'
    CLFLAGS_inverse = inv_map = {v[1:]: k for k, v in CLFLAGS.iteritems()}

    for line in lines:
        # Discard whitespace (space, tab)
        line = line.replace(" ", "")
        line = line.replace("\t", "")

        # Discard in-line comment and newline character
        line = line.split("#")[0].strip()

        # Find left-hand side of the line in arg. dict. and store value(s)
        argname = line.split("=")[0]
        argval = "=".join(line.split("=")[1:]).split(",")
        if argname in CLFLAGS_inverse.keys():
            if argval != ['']:
                if args[CLFLAGS_inverse[argname]] is not None:
                    print ("WARNING: Argument '{}' specified more than once. "
                           "The last specification will be used."
                           .format(argname))
                args[CLFLAGS_inverse[argname]] = argval
        else:
            print ("WARNING: {} is not a valid argument.".format(argname))

    # Set argument types for floating-point inputs
    for key in FLOAT_ARGS:
        try:
            args[key] = [float(val) for val in args[key]]
        except:
            raise TypeMismatchException("Invalid input for argument {}"
                                        .format(CLFLAGS[key]))
    # Set argument types for integer inputs
    for key in INT_ARGS:
        try:
            args[key] = [int(val) for val in args[key]]
        except:
            raise TypeMismatchException("Invalid input for argument {}"
                                        .format(CLFLAGS[key]))

    # Set argument types for boolean inputs
    for key in BOOL_ARGS:
        try:
            args[key] = [eval(str(val).title()) for val in args[key]]
        except:
            raise TypeMismatchException("Invalid input for argument {}"
                                        .format(CLFLAGS[key]))

    # Just like in parse_arguments, count the inputs and validate the integrity
    # of input counts here.
    excluded_keys = {"bids_dirs", "copy", "auto", "verbose", "config"}
    keycounts = {key: len(args[key]) for key in CLFLAGS.keys()
                 if (key not in excluded_keys)
                 and (args[key] is not None)}
    inputcount = max(keycounts.values())
    if len(set(keycounts.values())) != 1:
        print ("WARNING: The number of inputs is not the same for all "
               "arguments. This might result in different operations being "
               "performed on consecutive inputs. If the inequality is due "
               "to a missing input, the results will likely be corrupted "
               "after the missing input.")
        # DO NOT TERMINATE. The user might have wanted to perform different
        # operations that needed different inputs.
    else:
        args['inputcount'] = inputcount

    return args


def parse_arguments():
    """Sub-routine that understands command-line arguments and passes the
    information to the main program."""

    # Initialise argument dictionary (that will be returned)
    args = dict(zip(CLFLAGS.keys(), [None] * len(CLFLAGS)))

    # Check for a config file specification
    if argexist(CLFLAGS['config']):
        if argexist(CLFLAGS['config'], True):
            # If the config argument is set, discard all other arguments
            args['config'] = subarg(CLFLAGS['config'])
            # Except for -auto, that is necessary to bypass the first
            # user confirmation
            args['auto'] = argexist(CLFLAGS['auto'])
            return args
        else:
            raise MissingArgumentException("Missing path for config file.")

    # String-type input arguments
    for key in STRING_ARGS:
        if argexist(CLFLAGS[key]):
            if argexist(CLFLAGS[key], True):
                args[key] = subarg(CLFLAGS[key], default_value=None)
            else:
                raise MissingArgumentException("Missing specification for "
                                               "argument {}."
                                               .format(CLFLAGS[key]))
        else:
            # If not specified, take the default value as many times as the
            # number of inputs for other arguments. For this, the input counts
            # must be verified after all specified arguments have been read.
            pass

    # Floating-point input arguments
    for key in FLOAT_ARGS:
        if argexist(CLFLAGS[key]):
            if argexist(CLFLAGS[key], True):
                try:
                    args[key] = [float(val) for val in
                                 subarg(CLFLAGS[key], default_value=None)]
                except:
                    raise ValueError("Invalid input for argument {}."
                                     .format(CLFLAGS[key]))
            else:
                raise MissingArgumentException("Missing specification for "
                                               "argument {}."
                                               .format(CLFLAGS[key]))
        else:
            # If not specified, take the default value as many times as the
            # number of inputs for other arguments. For this, the input counts
            # must be verified after all specified arguments have been read.
            pass

    # Integer input arguments
    for key in INT_ARGS:
        if argexist(CLFLAGS[key]):
            if argexist(CLFLAGS[key], True):
                try:
                    args[key] = [int(val) for val in
                                 subarg(CLFLAGS[key], default_value=None)]
                except:
                    raise ValueError("Invalid input for argument {}."
                                     .format(CLFLAGS[key]))
            else:
                raise MissingArgumentException("Missing specification for "
                                               "argument {}."
                                               .format(CLFLAGS[key]))
        else:
            # If not specified, take the default value as many times as the
            # number of inputs for other arguments. For this, the input counts
            # must be verified after all specified arguments have been read.
            pass

    # Boolean-type input arguments
    for key in BOOL_ARGS:
        args[key] = argexist(CLFLAGS[key])

    # Count the input
    excluded_keys = {"bids_dirs", "copy", "auto", "verbose", "config"}
    keycounts = {key: len(args[key]) for key in CLFLAGS.keys()
                 if (key not in excluded_keys)
                 and (args[key] is not None)}
    inputcount = max(keycounts.values())
    if len(set(keycounts.values())) != 1:
        print ("WARNING: The number of inputs is not the same for all "
               "arguments. This might result in different operations being "
               "performed on consecutive inputs. If the inequality is due "
               "to a missing input, the results will likely be corrupted "
               "after the missing input.")
        # DO NOT TERMINATE. The user might have wanted to perform different
        # operations that needed different inputs.
    else:
        args['inputcount'] = inputcount

    # Add default values for all unspecified arguments.
    # NOTE: Unless all arguments are compulsory, this might force specifications
    # against the user's intent.
    # SPECIAL BEHAVIOUR:
    #   bids_dirs: explicit argument (must be mentioned even to be set to def.)
    for key in ARG_DEFAULTS.keys():
        if key not in EXPLICIT_ARGS:
            if args[key] is None:
                args[key] = ARG_DEFAULTS[key] * inputcount

    return args


def check_requirements(requirements, args_dictionary):
    if all([bool(args_dictionary[req]) for req in requirements]):
        return True
    else:
        return False


def task_selector(args):
    """This sub-routine decides based on the program input what operations
    should be performed."""

    # Define boolean task dictionary (this will be returned)
    tasks = OrderedDict(zip(zip(
        *sorted([(v, k) for (k,v) in TASK_ORDER.iteritems()]))[1],
                            [None] * len(TASK_LIST)))

    # Load FSL
    tasks['load_fsl'] = True

    # Create BIDS directories
    reqs = {"id", "bids_dirs"}
    tasks['create_bids_dirs'] = check_requirements(reqs, args)

    # Create scanner field map
    reqs = {"id", "fmag", "fphase", "echodiff", "fractint"}
    tasks['create_field_map'] = check_requirements(reqs, args)

    # Load scanner field map
    reqs = {"fmap"}
    tasks['load_field_map'] = check_requirements(reqs, args)
    # As loading a field map is easier than calculating it, let this task take
    # precedence.
    if tasks['load_field_map']:
        tasks['create_field_map'] = False

    # Copy structural scan to the respective BIDS directory
    reqs = {"id", "struct", "copy"}
    tasks['copy_structural_to_bids'] = check_requirements(reqs, args)

    # Copy single-echo functional scan and single-echo reference image to the
    # respective BIDS directory
    reqs = {"id", "single_echo", "sref", "stime", "copy"}
    tasks['copy_single_echo_to_bids'] = check_requirements(reqs, args)

    # Copy multi-echo functional scan and multi-echo reference image to the
    # respective BIDS directory
    reqs = {"id", "multi_echo", "mref", "mtime", "copy"}
    tasks['copy_multi_echo_to_bids'] = check_requirements(reqs, args)

    # Create cheating EV file
    # NOTE: We use FEAT on the single-echo data, so I set the requirements so.
    reqs = {"single_echo"}
    tasks['create_cheating_ev'] = check_requirements(reqs, args)

    # Pad single-echo functional image and single-echo reference image
    # (provides compatibility with motion correction)
    reqs = {"id", "single_echo", "sref"}
    tasks['pad_single_echo'] = check_requirements(reqs, args)

    # Pad multi-echo functional image and multi-echo reference image
    # (provides compatibility with motion correction)
    reqs = {"id", "multi_echo", "mref"}
    tasks['pad_multi_echo'] = check_requirements(reqs, args)

    # Run fsl_anat on the structural image
    reqs = {"id", "struct"}
    tasks['run_fsl_anat'] = check_requirements(reqs, args)

    # Load the results from a previous fsl_anat session
    reqs = {"anatdir"}
    tasks['load_fsl_anatdir'] = check_requirements(reqs, args)
    # Loading the results is easier than running fsl_anat again, so let this
    # task take precedence.
    if tasks['load_fsl_anatdir']:
        tasks['run_fsl_anat'] = False

    # Run FEAT analysis on the single-echo data
    # This requires the single-echo functional scan and the single-echo
    # reference image. Both have to be padded. It also requires the field map
    # (either it was loaded or calculated). It also requires the field-corrected
    # and brain-extracted structural scan (from fsl_anat). Finally, it requires
    # the cheating EV file.
    tasks['run_feat'] = tasks['pad_single_echo'] and \
                        (tasks['create_field_map'] or
                         tasks['load_field_map']) and \
                        (tasks['run_fsl_anat'] or
                         tasks['load_fsl_anatdir']) and \
                        tasks['create_cheating_ev']

    # Load the results of a previous FEAT session
    # NOTE: As we use FEAT for the single-echo data, I set the requirements so.
    reqs = {"sfeat"}
    tasks['load_featdir'] = check_requirements(reqs, args)
    # If the .feat directory is available from a previous FEAT session, use that
    # instead of running it again.
    if tasks['load_featdir']:
        tasks['run_feat'] = False

    # Load physiological signal recordings (from BioPac)
    tasks['load_biodata'] = check_requirements({"sbio"}, args) or \
                            check_requirements({"mbio"}, args)

    # Run analysis on the single-echo data
    # Beyond the obvious input files, the analysis requires the results from
    # fsl_anat and FEAT.
    reqs = {"id", "single_echo", "sref", "stime", "sfmin", "spval", "sconv",
            "sbio", "sfreq"}
    tasks['single_echo_analysis'] = check_requirements(reqs, args) and \
                                    (tasks['run_fsl_anat'] or
                                     tasks['load_fsl_anatdir']) and \
                                    (tasks['run_feat'] or
                                     tasks['load_featdir'])

    # Prepare multi-echo data (sorting echos and motion correction)
    # Beyond the obvious input files, the preparation requires the padded
    # version of the reference image.
    # NOTE: The Matlab script uses the single-echo reference image, which I
    # consider a mistake. I set the requirement to the multi-echo reference
    # image.
    tasks['prepare_multi_echo'] = tasks['pad_multi_echo']

    # Run analysis on the multi-echo data
    reqs = {"id", "multi_echo", "mref", "mtime", "mbio", "sfreq", "cseg"}
    tasks['multi_echo_analysis'] = check_requirements(reqs, args)

    return tasks


def summarize(args):
    """Sub-routine for printing the summary on the screen."""

    # For the following code segment, it is important to understand the data
    # structure being dealt with.
    #   1. Multiple config files can be imported from the command line
    #      (or a config file).
    #   2. A config file can import exactly one set of arguments.
    #   3. Using the command line, exactly one set of arguments can be set at
    #      once.
    #   4. A set of arguments may contain more than one value for each argument.
    #      If there are more than one entries for an argument, they are treated
    #      as separate inputs, as if the program was run multiple times with
    #      different singular inputs. The inputs are matched, i.e. two inputs in
    #      the different argument listings correspond to each other if they
    #      share the same index. Two two exceptions are: config and bids_dirs.
    #      When 'config' is specified, all other arguments are discarded from
    #      the command, except for 'auto'. Bids_dirs: since it is plural even
    #      when other arguments are singular, the entry for 'bids_dirs'
    #      universally applies for all lines of input in any set of arguments.
    #
    # For the above reasons, the parse_args and config_file_parser sub-routines
    # export lists for every single argument value.

    # CREATE LIST WITH SETS OF ARGUMENTS (arg_sets)
    # If the program is executed with the 'config' argument set, parse the
    # configuration files to retrieve a set of arguments per config file.
    if args['config']:
        # Explicit looping must be used instead of an elegant list
        # comprehension, because exceptions must be handled.
        arg_sets = []
        for _, config_file in enumerate(args['config']):
            try:
                arg_sets.append(config_file_parser(config_file))
            except IOError as exc:
                # Show error if any of the configuration files cannot be opened.
                # Skip the file and move on to the next one.
                print exc.message
                continue
        else:
            # Handling the case when all inputs were invalid
            if not arg_sets:
                raise NothingToDoException("No valid input was specified.")

    # If the program was executed with command-line arguments, there is only a
    # single set of arguments.
    else:
        # A list with one entry: a dictionary with list values
        arg_sets = [parse_arguments()]

    # BREAK DOWN EVERY SET OF ARGUMENTS INTO DICTIONARIES OF CORRESPONDING
    # SINGULAR ARGUMENTS (whose values are not lists anymore)

    # Initialise "the great big pool": a list of dictionaries
    corresponding_args_list = []

    for _, arg_set in enumerate(arg_sets):  # note the singular and the plural!

        # DETERMINE THE MAXIMUM NUMBER OF INPUTS FOR THE CURRENT SET OF
        # ARGUMENTS
        # Note that boolean arguments are excluded from this verification, as
        # well as the list of BIDS directories that should be created, as these
        # are fixed options for all inputs. As long as the config argument
        # overrides everything (except for the auto), it is a redundancy, but
        # the config argument was excluded here as well. Excluding the
        # None-valued arguments doesn't provide error checking for non-set
        # compulsory arguments. These must be tested by task_selector.
        excluded_keys = {"bids_dirs", "copy", "auto", "verbose", "config"}
        keycounts = {key: len(arg_set[key]) for key in CLFLAGS.keys()
                     if (key not in excluded_keys)
                     and (arg_set[key] is not None)}
        if len(set(keycounts.values())) != 1:
            print ("WARNING: The number of inputs is not the same for all "
                   "arguments. This might result in different operations being "
                   "performed on consecutive inputs. If the inequality is due "
                   "to a missing input, the results will likely be corrupted "
                   "after the missing input.")
            # DO NOT TERMINATE. The user might have wanted to perform different
            # operations that needed different inputs.
        inputcount = max(keycounts.values())

        # DO THE SEPARATION
        # Extract individual values from arguments, whose values are still
        # lists. Do this the required number of times and in the mean time
        # add every new dictionary of corresponding singular arguments to
        # the great big pool.
        for i in range(inputcount):
            # Create a copy to transfer the universal argument values
            current_corresponding_args = arg_set.copy()


            # Equate the length of different argument value listings by adding
            # None-s.
            for key in current_corresponding_args.keys():
                if (type(current_corresponding_args[key]) is list) and \
                        (len(current_corresponding_args[key]) == 1):
                    current_corresponding_args[key] = \
                        current_corresponding_args[key][0]
                if key not in excluded_keys:
                    if type(current_corresponding_args[key]) is list:
                        while len(current_corresponding_args[key]) < inputcount:
                            current_corresponding_args[key].append(None)

            # Extract the corresponding singular arguments using list
            # comprehension and add it to the great big pool of dictionaries of
            # corresponding arguments. IMPORTANT: keep the singular values as
            # lists to create compatibility with list-valued arguments
            # (bids_dirs).
            for key in current_corresponding_args.keys():
                if key not in excluded_keys:
                    if type(current_corresponding_args[key]) is list:
                        current_corresponding_args[key] = \
                            current_corresponding_args[key][i]
            corresponding_args_list.append(current_corresponding_args)

    # CREATE LIST OF TASK LISTS (corresponding to the dictionaries in args_list)
    print corresponding_args_list
    tasks_list = [task_selector(corresponding_args)
                 for _, corresponding_args
                 in enumerate(corresponding_args_list)]


    # DISPLAY SUMMARY IN INTERACTIVE MODE
    if args['auto']:
        # Display nothing when Interactive Mode is switched off
        pass
    else:
        # HEADER
        msg = "List of all operations ({}) waiting to be performed:"\
              .format(len(corresponding_args_list))
        print msg
        print "".join(['='] * len(msg))

        # BODY
        inputkey = "id"
        # inputkey = max(keycounts.iterkeys(), key=lambda k: keycounts[k])
        for i, corresponding_args in enumerate(corresponding_args_list):
            corresponding_args['inputkey'] = str(inputkey)
            print "Input:", corresponding_args[inputkey]
            print "\t" + "\n\t".join([key for key in tasks_list[i]
                                      if tasks_list[i][key]])

    return corresponding_args_list, tasks_list


def main():
    """Main program code. Operates through 5 steps:
        I. Read and validate input
        II. Summarize task and ask for user confirmation
        III. Perform analysis steps
        IV. Create output
        V. Report results"""

    # I. Read and validate input
    try:
        args = parse_arguments()
    except:
        # TODO: Placeholder for proper exception handling
        raise

    # II. Summarize task and ask for user confirmation
    try:
        args_list, tasks_list = summarize(args)
        # Bypass user confirmation when Interactive Mode is off.
        if not args['auto']:
            if not confirmed_to_proceed("\n{} operation(s) will be performed. "
                                        "Would you like to proceed? (y/n): "
                                        .format(len(args_list))):
                exit(1)
    except:
        # TODO: Placeholder for proper exception handling
        # This captures the NothingToDoException.
        raise

    # III. Perform steps
    # TODO: This is all-linear processing. Could be paralellised.
    for i, tasks in enumerate(tasks_list):
        current_args = args_list[i]
        # Convert relative paths to absolute paths. Use the current working
        # directory as a reference.
        current_args = csfmri_tasks.absolutise_paths(current_args)
        # Extract the subject label from sebjectID, since the latter was
        # allowed to be specified as a path
        current_args['label'] = csfmri_tasks.extract_subject_label(current_args)
        # Perform the current set of tasks by executing one after the other. In
        # case of an exception, try to continue the work as long as possible.

        # Just for debugging
        print current_args

        # Perform tasks
        for _, task in enumerate(tasks):
            # If the task execution value is True, run the task
            if tasks[task]:
                try:
                    getattr(csfmri_tasks, task)(current_args)
                except:
                    # TODO: Add proper exception handling
                    csfmri_tasks._status("ERROR while performing task {} on input "
                                         "No. {} ('{}')".format(str(task), i,
                                         current_args[current_args['inputkey']]),
                                         current_args)
                    continue

    # IV. Create output

    # V. Report results


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

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

"""This Python module contains certain utilities that are frequently needed in
various project scripts. Be aware that changing the content of this file
requires a thorough understanding of the structure and content of the project
files, as it might result in unwanted behaviour in multiple applications.

This script is not intended to be directly executed."""

# IMPORTS

import os
from cl_interface import *
from csfmri_exceptions import *


# DEFINITIONS AND CODE

# Supported file formats and extensions
FORMATS = {'NIFTI': 'nii',
           'NIFTI_GZ': 'nii.gz'}

# Maximum number of parallel processes
N_CPU = 4

# Default output tag
DEFAULT_OUTPUT_TAG = "_out"


def matched_input_validation(**kwargs):
    """Parsing command-line arguments often involves reading lists of input
    (and output) files that are related to each other (e.g. same subject).
    Therefore, it is necessary to check if the lists are of equal size and all
    files are supported. This sub-routine provides a template for these
    operations."""

    # Decompose function arguments and check if they exist
    try:
        inputs = kwargs['inputs']
        assert inputs
    except:
        raise MissingArgumentException("Matched input validation: Inputs must "
                                       "be specified.")
    try:
        supported_formats = kwargs['supported_formats']
        assert supported_formats
    except:
        raise MissingArgumentException("Matched input validation: Supported "
                                       "formats must be specified.")
    try:
        interactive = bool(kwargs['interactive'])
    except:
        interactive = True  # The interactive option is set to True by default.

    # Check if all inputs have the same data type (list is expected)
    input_types = set([type(ipt) for ipt in inputs])
    if len(input_types) != 1:
        raise TypeMismatchException("Input types are not compatible.")
    # Enforce list type
    if input_types.pop() != list:
        inputs = [[ipt] for ipt in inputs]
    filtered_inputs = inputs

    # Check if the counts for each input are equal
    number_of_inputs = set([len(ipt) for ipt in inputs])
    if len(number_of_inputs) != 1:
        raise CountsMismatchException("Input counts do not match.")
    else:
        number_of_inputs = number_of_inputs.pop()

    # Check if any of the input files is unsupported (by file extension)
    unsupported_inputs = []
    for input_index, input_files in enumerate(zip(*inputs)):
        for input_file in input_files:
            if all([os.path.split(input_file)[1].lower().index(fmt) == -1
                    for fmt in supported_formats]):
                unsupported_inputs.append((input_index, input_file))
    if unsupported_inputs:
        errmsg = ("The following input(s) have unsupported format:\n"
                  + "\n".join(zip(*unsupported_inputs)[1]))
        if interactive:
            print errmsg
            print ("Would you like to discard these? (y/n): ")
            if confirmed_to_proceed():
                indices_to_discard = set(zip(*unsupported_inputs)[0])
                zipped_inputs = zip(*inputs)
                filtered_inputs = [zipped_inputs[i]
                                   for i in range(number_of_inputs)
                                   if not (i in indices_to_discard)]
                # Update the number of inputs
                print ("The number of discarded inputs: {}."
                       .format(len(indices_to_discard)))
                number_of_inputs = number_of_inputs - len(indices_to_discard)
                if number_of_inputs <= 0:
                    raise NothingToDoException("No input left.")
            else:
                raise UnsupportedFormatException(errmsg)
        else:
            raise UnsupportedFormatException(errmsg)

    return zip(*filtered_inputs)


def guess_output_from_input(inputfiles, basename=None, tag=None, subdir=None):
    """Create a list of output file specifications based on the input. The
    path is the result of the input path and an optionally specified
    subdirectory. The file name is based on an optionally specified base name.
    When no base name is specified, the input file name is used with the
    specified or the default output tag. The extension is copied from the input.
    Please note that the resultant output specifications must be validated
    before saving. It is assumed that all inputs are files and not directories.
    """

    # Check if the function arguments are correct
    if type(inputfiles) != list:
        inputfiles = [inputfiles]
    if not subdir:
        subdir = ""

    output_list = []
    for input_file in inputfiles:
        # Guess location
        input_path, input_name = os.path.split(input_file)

        # Add optionally specified subdir relative to the source directory
        # (This is an easy way to allow ../../ type referencing in subdir.)
        current_dir = os.getcwd()
        os.chdir(input_path)
        output_path = input_path
        try:
            os.makedirs(path=subdir, exist_ok=True)
            os.chdir(subdir)
            output_path = os.getcwd()
            os.chdir(current_dir)
        except:
            raise GenericIOException("Guess output: the specified "
                                     "sub-directory could not be created.")

        # Guess extension
        for input_extension in FORMATS.values():
            if input_name.lower().find(input_extension) != -1:
                break

        # Guess name
        if not basename:
            output_name = input_name[:input_name.lower()
                                     .index(input_extension)-1] # -1 removes dot
            if tag:
                output_name = output_name + tag
            else:
                # Avoid overwriting input file when no subdirectory is specified
                if not subdir:
                    output_name = output_name + DEFAULT_OUTPUT_TAG
        else:
            output_name = basename
            if tag:
                output_name = output_name + tag

        # Add output specification to the list
        # Duplicates might be present in the list
        output_list.append(os.path.join(output_path, subdir, output_name,
                                        "."+input_extension))

    return output_list



# Display information if accidentally executed
if __name__ == "__main__":
    print ("{} is not intended for execution."
           .format(os.path.basename(__file__)))
    exit(0)
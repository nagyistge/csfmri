Readme for the Cardio-Synchronous fMRI (CSfMRI) Project's scripts

Script files:
-------------

main.py -config <configfile.conf>
   Main executable. Used with the given arguments above, it reads a configuration file and decides what to do.

csfmri_tasks.py
   Implementation of the individual tasks, that are called executed by main.py. Tasks include creating directory tree, creating or loading field map, running fsl_anat and feat (or loading the respective directories from a previous run), padding volumes, single-echo analysis, multi-echo analysis, etc.

csfmri_exceptions.py
   A few customised exception classes. Mainly for future use.

cl_interface.py
   Custom command line interface for reading the program arguments and interacting with the user.

fsl_interface.py
   Intended as an extensive Python interface library for FSL, but currently limited to locating the FSL installation directory.

All further scripts have become obsolete and are no longer required for the analysis. Some of them however were intended to be usable on their own right (e.g. for creating the subject's directory tree or the field map).

Auxiliary files:
----------------

feat.fsf
   A template configuration file for FEAT. This is copied and customised when the program needs to run FEAT.

single_analysis.conf
multi_analysis.conf
full_analysis.conf
   This are the configuration files for running single-echo analysis, multi-echo analysis and both, respectively.

Single_echo_timing.txt
Multi_echo_timing.txt
   Acquisition timing descriptor files. They contain the information on TR, TE, and slice timing plus the number of repetitions until steady state. The PADDED option tells whether the supplied slice timing information refers to the padded volume or it needs to be padded in the motion correction step.

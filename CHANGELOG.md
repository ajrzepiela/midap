# Changelog (template)

## Legend for changelogs

- Feature : something that you couldn’t do before.
- Efficiency : an existing feature now may not require as much computation or memory.
- Fix : something that previously didn’t work as documentated – or according to reasonable expectations – should now work.

## Version number

Date

Feature: 
- ...

Efficiency:
- ...

Fix:
- ...

## [0.0.5]

2022-10-13

Feature:
- Integrated M1 workflow into the main script such that one can run everything from one script
- Added support for Euler setup
- Add instruction for both to README, for now Euler only with LSF
- Removed matlab suppersegger and modules

Fix:
- Open CV dependency was not in the requirements

## [0.0.4]

2022-10-10

Feature:
- Changes the `src` folder to `midap` and transformed the repo into packages
- Added modularity support for chamber cutting and cell segmentation

Fix:
- Minor fix in tracking to ensure that trackID_list is at least as long as global_IDs

## [0.0.3]

2022-10-05

Feature:
- Pipeline supports running PH + N Channels via comma separated list of channels in the GUI
- Changed backend of matplolib to be compatible with CentOS clusters
- Added logging utils for standardized logging
- Started moving scripts to apps folder

Fix:
- Added quotes to support file names with spaces
- Fix bug where local bash variables with interpolated commands (`local VAR=$(...)`) are not trapped when written on one line
- Set -maxdepth of find command in copy_files to 1 to avoid finding more than one file in the datasets
- Catch Error if no identifier was found
- Added an exeption if error was out of callstack

Efficiency:
- Cleanup of files in apps folder and their related imports in src
- Preparation of cut chambers and segmentation for user defined subclasses



## [0.0.2]

2022-09-16

Feature:
- `set_parameters.py` puts a header into `setting.sh` containing the current date, time and the git hash of the repo
- Additional metadata like the cutout corners or chosen models are saved into `settings.sh`
- `settings.sh` is copied to each data folder in the FAMILY\_MACHINE setting for reproducibility
- New argument `--loglevel` for the bash script to choose verbosity of output
- `--restart` flag now accepting an optional path to restart from other directories

Fix:
- Tensorflow output is suppressed by default, change `TF_CPP_MIN_LOG_LEVEL` in bash script to increase
- Variables that are already set in `settings.sh` are updated instead of appended at the end
- Moved last copy settings inside tracking function to avoid unintened overwrite for restarts
- Fix bug in cut images where an alignment offset of 0 caused a crash

## [0.0.1]

2022-09-13

Feature:
- bash script `run_pipeline_test.sh` accecpting inputs
- pipeline can be restarted from checkpoints
- bash script has logging

Fix:
- GUI elements `restrict_frames.py` and `restrict_frames.py` exit with code 1 when "Cancel" or "X" button is pressed.



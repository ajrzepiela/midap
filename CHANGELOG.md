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

## [0.0.1]

2022-09-13

Feature:
- bash script `run_pipeline_test.sh` accecpting inputs
- pipeline can be restarted from checkpoints
- bash script has logging

Fix:
- GUI elements `restrict_frames.py` and `restrict_frames.py` exit with code 1 when "Cancel" or "X" button is pressed.



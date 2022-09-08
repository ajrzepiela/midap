#!/bin/bash

# Argument Parsing
##################

help()
{
  echo "Runs the cell segmentation and tracking pipeline"
  echo 
  echo "Syntax: run_pipeline_checkpoints.sh [options]"
  echo 
  echo "Options:"
  echo " -h, --help    Display this help"
  echo " --restart     Restart pipeline from log file"
  echo " --headless   Run pipeline in headless mode (no GUI)"
  echo 
  exit 2
}


while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--help)
      help
      ;;
    --restart)
      RESTART="True"
      shift # past argument
      ;;
    --headless)
      HEADLESS="True"
      shift # past argument
      ;;
    -*|--*)
      echo "Unknown option $1"
      echo
      help
      exit 1
      ;;
    *)
      echo "No positional arguments accepted!"
      echo 
      help
      exit 1
      ;;
  esac
done


# Logging
#########

# set verbose level
__VERBOSE=7

declare -A LOG_LEVELS
# https://en.wikipedia.org/wiki/Syslog#Severity_level
LOG_LEVELS=([0]="emerg" [1]="alert" [2]="crit" [3]="err" [4]="warning" [5]="notice" [6]="info" [7]="debug")
function .log () {
  local LEVEL=${1}
  shift
  if [ ${__VERBOSE} -ge ${LEVEL} ]; then
    echo "[${LOG_LEVELS[$LEVEL]}]" "$@"
  fi
}

# Checkpointing
###############

# We clear the log if we are successful
trap 'clear_log' EXIT
# If there is an error or interrupt we log
trap 'log_checkpoint $current_func' ERR SIGINT SIGHUP SIGKILL SIGTERM

CHECKLOG=checkpoints.log

clear_log() {
  # make sure the last exit code was 0
  if [ $? -eq 0 ] ; then
    rm -f "$CHECKLOG"
  fi
}

log_checkpoint() {
  # Print fail and log
  .log 2 "Error while running: $1"
  echo "$1" > $CHECKLOG
  exit 1
}

retry() {
  # transfer the function name into the current current_func
  current_func=$1
  # If there is no checkpoint file, there is nothing to do
  [ ! -f $CHECKLOG ] && return 0
  # if we have a checkpoint file we return 0 if it contains the current function
  if grep -q "$1" "$CHECKLOG"; then
    .log 6 "retry $1"; rm "$CHECKLOG"; return 0
  else
    .log 6 "skip $1"; return 1
  fi
}

# Pipeline
##########

set_parameters() {
  # Start GUI only if we are not in headless mode and no retry
  if [ "$HEADLESS" != "True" ] && retry ${FUNCNAME[0]}; then
    .log 7 "Starting up the GUI"
    python set_parameters.py
  fi
  # In case of headless or checkpoint we just source the settings
  .log 7 "Sourcing parameters 'settings.sh'"
  source settings.sh
}




# Callstack
###########

# If the restart option is not set, we delete the log file
if [ "$RESTART" != "True" ]; then
  .log 7 "Clearing checkpoing file"
  clear_log
fi

# set the parameters
set_parameters

# Family MAchine case
if [[ $DATA_TYPE == "FAMILY_MACHINE" ]]; then
  .log 7 "Working in: $PATH_FOLDER"
 
  # extract different positions from one dataset
  POSITIONS=()
  for i in $PATH_FOLDER*.$FILE_TYPE; do
    POS=$(echo $i | grep -Eo "${POS_IDENTIFIER}[0-9]+")
    POSITIONS+=($POS)
  done
  # Keep only unique 
  POS_UNIQ=($(printf "%s\n" "${POSITIONS[@]}" | sort -u));

  .log 7 "Extraced Identifiers: ${POS_UNIQ[@]}"

  # cycle through all identifiers
  for POS in "${POS_UNIQ[@]}"; do
    .log 7 "Starting with: ${POS}"

    

  done
 
 
fi

exit 0

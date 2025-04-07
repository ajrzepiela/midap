#!/bin/bash
#This script takes a input folder path, and submits jobs for midap headless_cluster at each position of it individually

# Exit if any command fails
set -e

# Check if input folder is provided
if [ -z "$1" ]; then
  echo "Usage: $0 PATH/TO/FOLDER"
  exit 1
fi

FOLDER="$1"
SETTINGS_FILE="$FOLDER/settings.ini"

# Check if settings.ini exists
if [ ! -f "$SETTINGS_FILE" ]; then
  echo "Error: settings.ini not found in $FOLDER"
  exit 1
fi

# Read the IdentifierFound line
IDENTIFIER_LINE=$(grep -m 1 '^IdentifierFound' "$SETTINGS_FILE")
if [ -z "$IDENTIFIER_LINE" ]; then
  echo "Error: IdentifierFound line not found in settings.ini"
  exit 1
fi

# Extract positions (removes spaces and splits by comma)
POSITIONS=$(echo "$IDENTIFIER_LINE" | cut -d'=' -f2 | tr -d ' ' | tr ',' '\n')

# Submit a job for each position
for POS in $POSITIONS; do
  echo "Submitting job for position: $POS"
  CMD="Command submitted: midap --headless_cluster $FOLDER $POS"
  echo "$CMD"
  sbatch --ntasks=1 --cpus-per-task=4 --time=24:00:00 --mem-per-cpu=4096 --wrap="midap --headless_cluster $FOLDER $POS"
done

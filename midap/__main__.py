import pkg_resources
import argparse
import shutil
import sys
import os
import re

import numpy as np

from shutil import copyfile
from pathlib import Path
from glob import glob

def run_module(args=None):
    """
    Runs the module, i.e. the cell segmentation and tracking pipeline.
    :param args: arguments to parse, defaults to reading in the arguments from the standard input (command line)
    """

    # get the args
    if args is None:
        args = sys.argv[1:]

    # argument parsing
    description = "Runs the cell segmentation and tracking pipeline."
    parser = argparse.ArgumentParser(description=description, add_help=True,
                                     usage="python -m midap [-h] [--restart [RESTART]] [--headless] "
                                           "[--loglevel LOGLEVEL] [--cpu_only] [--create_config]")
    # This arge is default if the flag is not set, it is const if it is set without arg, and it is the arg if provided
    parser.add_argument("--restart", nargs="?", default=None, const='.', type=str,
                        help="Restart pipeline from log file. If a path is specified the checkpoint and settings file "
                             "will be restored from the path, otherwise the current working directory is searched.")
    parser.add_argument("--headless", action="store_true",
                        help="Run pipeline in headless mode, ALL parameters have to be set prior in a config file.")
    parser.add_argument("--loglevel", type=int, default=7,
                        help="Set logging level of script (0-7), defaults to 7 (max log)")
    parser.add_argument("--cpu_only", action="store_true",
                        help="Sets CUDA_VISIBLE_DEVICES to -1 which will cause most! applications to use CPU only.")
    parser.add_argument("--create_config", action="store_true",
                        help="If this flag is set, all other arguments will be ignored and a 'settings.ini' config "
                             "file is generated in the current working directory. This option is meant generate "
                             "config file templates for the '--headless' mode. Note that this will overwrite "
                             "if a file already exists.")

    # parsing
    args = parser.parse_args(args)

    # Some constants or conventions
    config_file = "settings.ini"
    check_file = "checkpoints.log"
    raw_im_folder = "raw_im"
    cut_im_folder = "cut_im"
    seg_im_folder = "seg_im"
    track_folder = "track_output"

    # Argument handling
    ###################

    # we do the local imports here to intercept TF import
    if args.cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = -1
    # supress TF blurp
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # set the global loglevel
    os.environ["__VERBOSE"] = str(args.loglevel)

    # create a logger
    from .utils import get_logger
    logger = get_logger("MIDAP", args.loglevel)

    # print the version
    version = pkg_resources.require("midap")[0].version
    logger.info(f"Running MIDAP version: {version}")

    # imports
    logger.info(f"Importing all dependencies...")
    from .checkpoint import Checkpoint, CheckpointManager
    from .config import Config
    from .apps import init_GUI
    logger.info("Done!")

    # create a config file if requested and exit
    if args.create_config:
        config = Config()
        config.to_file("settings.ini", overwrite=True)
        return 0

    # check if we are restarting
    restart = False
    if args.restart is not None:
        # we set the restart flag
        restart = True

        # we search for the checkpoint file (recursive subdirectory search)
        checkpoints = glob(os.path.join(args.restart, "**", check_file), recursive=True)

        # exception handling
        if len(checkpoints) == 0:
            raise FileNotFoundError(f"No checkpoint found in the restart directory: {args.restart}")
        if len(checkpoints) > 1:
            raise ValueError(f"Multiple checkpoints found in the restart directory: {args.restart}")

        # we extract the checkpoing
        prev_checkpoint = Path(checkpoints[0]).absolute()
        logger.info(f"Found checkpoint: {prev_checkpoint}")

        # now we get the corresponding config file
        prev_config = prev_checkpoint.parent.joinpath(config_file)
        if not prev_config.exists():
            raise FileNotFoundError(f"No corresponding config file found: {prev_config}")
        logger.info(f"Found corresponding config file: {prev_config}")

        # we copy the files to the current working directory (if necessary)
        try:
            copyfile(prev_config, config_file)
            copyfile(prev_checkpoint, check_file)
        except shutil.SameFileError:
            pass

        # now we create the config and checkpoint (we do a full check if we are in headless mode)
        config = Config.from_file(config_file, full_check=args.headless)
        checkpoint = Checkpoint.from_file(check_file)

    # since we do a full check above if we have headless mode and restart this is an elif
    elif args.headless:
        # read the config from the working directory
        logger.info("Running in headless mode, checking config file...")
        config = Config.from_file(config_file, full_check=True)
        checkpoint = Checkpoint(check_file)

        # we create a checkpoint with the current config
        # Note that this is a dummy checkpoint such that we can use the --restart flag in the worst case
        checkpoint.set_state(state="InitGUI", flush=True)

    # we are not restarting nor are we in headless mode
    else:
        logger.info("Starting up initial GUI...")
        # start the GUI
        init_GUI.main(config_file=config_file)
        # load in the config and create a checkpoint
        config = Config.from_file(fname=config_file, full_check=False)
        checkpoint = Checkpoint(check_file)

        # we create a checkpoint with the current config
        # Note that this is a dummy checkpoint such that we can use the --restart flag in the worst case
        checkpoint.set_state(state="InitGUI", flush=True)

    # Setup
    #######

    # we cycle through all pos identifiers
    for identifier in config.getlist("General", "IdentifierFound"):
        # read out what we need to do
        run_segmentation = config.get(identifier, "RunOption").lower() in ['both', 'segmentation']
        run_tracking = config.get(identifier, "RunOption").lower() in ['both', 'tracking']
        # get the current base folder
        base_path = Path(config.get("General", "FolderPath"))

        # stuff we do for the segmentation
        if run_segmentation:
            # define all the current paths
            current_path = base_path.joinpath(identifier)

            # setup all the directories
            with CheckpointManager(restart=restart, checkpoint=checkpoint, config=config, state="SetupDirs",
                                   identifier=identifier, copy_path=current_path) as checker:
                # check to skip
                checker.check()

                # remove the folder if it exists
                if current_path.exists():
                    shutil.rmtree(current_path, ignore_errors=False)

                # we create all the necessary directories
                current_path.mkdir(parents=True)

                # channel directories
                for channel in config.getlist(identifier, "Channels"):
                    current_path.joinpath(channel, raw_im_folder).mkdir(parents=True)
                    current_path.joinpath(channel, cut_im_folder).mkdir(parents=True)
                    current_path.joinpath(channel, seg_im_folder).mkdir(parents=True)
                    current_path.joinpath(channel, track_folder).mkdir(parents=True)













# main routine
if __name__ == "__main__":
    run_module()

import argparse
import os
import shutil
import sys
from glob import glob
from pathlib import Path
from shutil import copyfile

import pkg_resources


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
                                     usage="midap [-h] [--restart [RESTART]] [--headless] "
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
    parser.add_argument("--prepare_config_cluster", action="store_true",
                    help="This option is meant to generate a config file and the output folder structure for a "
                            "dataset. The output folder is decompressed and can be uploaded to the cluster to "
                            "continue the pipeline in the '--headless' mode. Note that this will overwrite "
                            "if a file already exists.")

    # parsing
    args = parser.parse_args(args)

    # Some constants or conventions
    config_file = "settings.ini"
    check_file = "checkpoints.log"

    # Argument handling
    ###################

    # we do the local imports here to intercept TF import
    if args.cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # supress TF blurp
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # set the global loglevel
    os.environ["__VERBOSE"] = str(args.loglevel)

    # create a logger
    from midap.utils import get_logger
    logger = get_logger("MIDAP", args.loglevel)

    # print the version
    version = pkg_resources.require("midap")[0].version
    logger.info(f"Running MIDAP version: {version}")
    
    # imports
    logger.info(f"Importing all dependencies...")
    # Note, some of these dependencies are only used in the individual modules, but we import them here anyway
    # to avoid loading times later
    from midap.checkpoint import Checkpoint, CheckpointManager
    from midap.config import Config
    from midap.apps import download_files, init_GUI, split_frames, cut_chamber, segment_cells, segment_analysis, track_cells
    from midap.main_family_machine import run_family_machine
    from midap.main_mother_machine import run_mother_machine
    logger.info("Done!")

    # Download the files if necessary
    logger.info(f"Checking necessary files...")
    download_files.main(args=[])
    logger.info("Done!")

    # create a config file if requested and exit
    if args.create_config:
        config = Config(fname="settings.ini")
        config.to_file(overwrite=True)
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

    # we are not restarting nor are we in headless mode
    else:
        logger.info("Starting up initial GUI...")
        # start the GUI
        init_GUI.main(config_file=config_file)
        # load in the config and create a checkpoint
        config = Config.from_file(fname=config_file, full_check=False)
        checkpoint = Checkpoint(check_file)

    # run the pipeline
    if config.get("General", "DataType") == "Family_Machine":
        run_family_machine(config=config, checkpoint=checkpoint, main_args=args, logger=logger, restart=restart)
    elif config.get("General", "DataType") == "Mother_Machine":
        run_mother_machine(config=config, checkpoint=checkpoint, main_args=args, logger=logger, restart=restart)
    else:
        raise ValueError(f"Unknown DataType: {config.get('General', 'DataType')}")


# main routine
if __name__ == "__main__":
    run_module()

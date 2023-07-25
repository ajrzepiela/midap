import shutil
import sys
from pathlib import Path
from shutil import copyfile

import numpy as np

from midap.apps import split_frames, cut_chamber, segment_cells, segment_analysis, track_cells
from midap.checkpoint import CheckpointManager

def run_mother_machine(config, checkpoint, main_args, logger, restart=False):
    """
    This function runs the mother machine.
    :param config: The config object to use
    :param checkpoint: The checkpoint object to use
    :param main_args: The args from the main function
    :param logger: The logger object to use
    :param restart: If we are in restart mode
    """

    # folder names
    raw_im_folder = "raw_im"
    cut_im_folder = "cut_im"
    seg_im_folder = "seg_im"
    seg_im_bin_folder = "seg_im_bin"
    track_folder = "track_output"

    # get the current base folder
    base_path = Path(config.get("General", "FolderPath"))

    # we cycle through all pos identifiers
    for identifier in config.getlist("General", "IdentifierFound"):
        # read out what we need to do
        run_segmentation = config.get(identifier, "RunOption").lower() in ['both', 'segmentation']
        # current path of the identifier
        current_path = base_path.joinpath(identifier)

        # stuff we do for the segmentation
        if run_segmentation:

            # setup all the directories
            with CheckpointManager(restart=restart, checkpoint=checkpoint, config=config, state="SetupDirs",
                                   identifier=identifier) as checker:
                # check to skip
                checker.check()

                logger.info(f"Generating folder structure for {identifier}")

                # remove the folder if it exists
                if current_path.exists():
                    shutil.rmtree(current_path, ignore_errors=False)

                # we create all the necessary directories
                current_path.mkdir(parents=True)

                # channel directories (only the raw images here, the rest is per chamber)
                for channel in config.getlist(identifier, "Channels"):
                    current_path.joinpath(channel, raw_im_folder).mkdir(parents=True)

            # copy the files
            with CheckpointManager(restart=restart, checkpoint=checkpoint, config=config, state="CopyFiles",
                                   identifier=identifier, copy_path=current_path) as checker:
                # check to skip
                checker.check()

                logger.info(f"Copying files for {identifier}")

                # we get all the files in the base bath that match
                file_ext = config.get("General", "FileType")
                if file_ext == "ome.tif":
                    files = base_path.glob(f"*{identifier}*/**/*.ome.tif")
                else:
                    files = base_path.glob(f"*{identifier}*.{file_ext}")
                for fname in files:
                    for channel in config.getlist(identifier, "Channels"):
                        if channel in fname.stem:
                            logger.info(f"Copying '{fname.name}'...")
                            copyfile(fname, current_path.joinpath(channel, fname.name))

            # This is just to fill in the config file, i.e. split files 2 frames, get corners, etc
            ######################################################################################

            # split frames
            with CheckpointManager(restart=restart, checkpoint=checkpoint, config=config, state="SplitFramesInit",
                                   identifier=identifier, copy_path=current_path) as checker:
                # check to skip
                checker.check()

                logger.info(f"Splitting test frames for {identifier}")

                # split the frames for all channels
                file_ext = config.get("General", "FileType")
                for channel in config.getlist(identifier, "Channels"):
                    paths = list(current_path.joinpath(channel).glob(f"*.{file_ext}"))
                    if len(paths) == 0:
                        raise FileNotFoundError(f"No file of the type '.{file_ext}' exists for channel {channel}")
                    if len(paths) > 1:
                        raise FileExistsError(f"More than one file of the type '.{file_ext}' "
                                              f"exists for channel {channel}")

                    # we only get the first frame and the mid frame
                    first_frame = config.getint(identifier, "StartFrame")
                    mid_frame = int(0.5*(first_frame + config.getint(identifier, "EndFrame")))
                    frames = np.unique([first_frame, mid_frame])
                    split_frames.main(path=paths[0], save_dir=current_path.joinpath(channel, raw_im_folder),
                                      frames=frames,
                                      deconv=config.get(identifier, "Deconvolution"),
                                      loglevel=main_args.loglevel)

            # cut chamber and images
            with CheckpointManager(restart=restart, checkpoint=checkpoint, config=config, state="CutFramesInit",
                                   identifier=identifier, copy_path=current_path) as checker:
                # check to skip
                checker.check()

                logger.info(f"Cutting test frames for {identifier}")

                # get the paths
                paths = [current_path.joinpath(channel, raw_im_folder)
                         for channel in config.getlist(identifier, "Channels")]

                # Do the init cutouts
                if config.get(identifier, "Corners") == "None" or config.get(identifier, "Offsets") == "None":
                    corners = None
                    offsets = None
                else:
                    corners = tuple([int(corner) for corner in config.getlist(identifier, "Corners")])
                    offsets = tuple([int(offset) for offset in config.getlist(identifier, "Offsets")])
                cut_corners, offsets = cut_chamber.main(channel=paths,
                                                        cutout_class=config.get(identifier, "CutImgClass"),
                                                        corners=corners,
                                                        offsets=offsets)

                # save the corners if necessary
                if corners is None or offsets is None:
                    corners = f"{cut_corners[0]},{cut_corners[1]},{cut_corners[2]},{cut_corners[3]}"
                    offsets = ",".join([str(offset) for offset in offsets])
                    config.set(identifier, "Corners", corners)
                    config.set(identifier, "Offsets", offsets)
                    config.to_file()
                raise NotImplementedError("This is not implemented yet")


            # select the networks
            with CheckpointManager(restart=restart, checkpoint=checkpoint, config=config, state="SegmentationInit",
                                   identifier=identifier, copy_path=current_path) as checker:
                # check to skip
                checker.check()

                logger.info(f"Segmenting test frames for {identifier}...")

                # cycle through all channels
                for num, channel in enumerate(config.getlist(identifier, "Channels")):
                    # The phase channel is always the first
                    if num == 0 and not config.getboolean(identifier, "PhaseSegmentation"):
                        continue

                    # get the current model weight (if defined)
                    model_weights = config.get(identifier, f"ModelWeights_{channel}", fallback=None)

                    # run the selector
                    segmentation_class = config.get(identifier, "SegmentationClass")
                    if segmentation_class == "OmniSegmentation":
                        path_model_weights = Path(__file__).parent.parent.joinpath("model_weights",
                                                                                   "model_weights_omni")
                    else:
                        raise ValueError(f"Unknown segmentation class {segmentation_class}")
                    weights = segment_cells.main(path_model_weights=path_model_weights, path_pos=current_path,
                                                 path_channel=channel, postprocessing=True, clean_border=config.get(identifier, "RemoveBorder"), network_name=model_weights,
                                                 segmentation_class=segmentation_class, just_select=True,
                                                 img_threshold=config.getfloat(identifier, "ImgThreshold"))

                    # save to config
                    if model_weights is None:
                        config.set(identifier, f"ModelWeights_{channel}", weights)
                        config.to_file()

                raise NotImplementedError("This is not implemented yet")


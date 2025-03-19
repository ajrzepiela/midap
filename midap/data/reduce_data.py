
import os
import tifffile as tiff

from midap.utils import get_logger

loglevel = 7
logger = get_logger(__file__, loglevel)

def filter_tiff_stack(input_file, output_file, from_idx, to_idx):
    """
    Filters a multi-page TIFF file to include only images between from_idx and to_idx.
    :param input_file: string with input file path value
    :param output_file: string with output file path value
    :param from_idx: integer value of the start index from which slices should be saved
    :param to_idx: integer value of the end index upto which slices should be saved
    """
    with tiff.TiffFile(input_file) as tif:
        images = tif.asarray()
    if from_idx < 0 or to_idx >= len(images) or from_idx > to_idx:
        raise ValueError("Invalid from/to indices")
    selected_images = images[from_idx:to_idx+1]
    tiff.imwrite(output_file, selected_images)
    logger.info(f"Saved {output_file} with slices {from_idx} to {to_idx}.")

def filter_data_set(input_folder, output_folder, from_idx, to_idx):
    """
    for a data set of multiple tiff files in a folder (input_folder), creates a copy of reduced complexity (output_folder). 
    within this copy, only z-stacks between the from_idx and the to_idx will be included
    :param input_folder: string with input folder path value
    :param output_folder: string with output folder path value
    :param from_idx: integer value of the start index from which slices should be saved
    :param to_idx: integer value of the end index upto which slices should be saved
    """
    if not os.path.isdir(input_folder):
        raise FileNotFoundError("Invalid input folder")
    files = os.listdir(input_folder)
    if not os.path.isdir(output_folder):
        logger.info(f"Creating output folder at location {output_folder}")
        os.mkdir(output_folder)
    logger.info(f"Initializing data cuting from source {input_folder} with destination {output_folder}")
    for f in files:
        filter_tiff_stack(os.path.join(input_folder,f), os.path.join(output_folder,f),from_idx,to_idx)


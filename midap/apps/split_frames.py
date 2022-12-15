from tqdm import tqdm
from scipy.io import loadmat
from skimage.restoration import richardson_lucy
from skimage import io
from typing import Union, Literal
from pathlib import Path

from midap.utils import get_logger

def main(path: Union[str,Path], save_dir: Union[str,Path], start_frame: int, end_frame: int,
         deconv: Literal["deconv_family_machine", "deconv_well", "no_deconv"], loglevel=7):
    """
    Splits the frames of a given file and saves it in the save dir
    :param path: Path to the file to split the frames
    :param start_frame: Number of the start frame (inclusive)
    :param end_frame: Number of the end frame (exclusive)
    :param deconv: A literal used for the deconvolution
    :param loglevel: The loglevel of the script from 0 (no output) to 7
    """

    # logging
    logger = get_logger(__file__, loglevel)
    logger.info(f"Splitting frames of: {path}")

    # paths
    path = Path(path)
    raw_filename = path.stem
    save_dir = Path(save_dir)

    # loop over tif/tiff-stack to extract single frames and deconvolve them if wanted
    if deconv == "deconv_family_machine":
        logger.debug("Running deconv for family machine.")
        psf = loadmat(Path(__file__).parent.parent.parent.joinpath("psf", "PSFmme.mat"))['PSF']
        deconvolution = True
    elif deconv == "deconv_well":
        logger.debug("Running deconv for well.")
        psf = io.imread(Path(__file__).parent.parent.parent.joinpath("psf", "PSF_BornWolf.tif"))[5,:,:]
        deconvolution = True
    else:
        logger.debug("No deconv selected")
        deconvolution = False

    # split the frames
    logger.info("Splitting frames...")
    stack = io.imread(path)[int(start_frame):int(end_frame)]
    for ix, frame in enumerate(tqdm(stack)):
        if deconvolution:
            deconvoluted = richardson_lucy(frame, psf, iterations=10, clip=False)
            io.imsave(save_dir.joinpath(f"{raw_filename}_frame{ix + start_frame:03d}_deconv.png"),
                      deconvoluted, check_contrast=False)
        else:
            io.imsave(save_dir.joinpath(f"{raw_filename}_frame{ix + start_frame:03d}.png"),
                      frame, check_contrast=False)


if __name__ == "__main__":
    # argument parsing
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to the file of which the frames should be split.")
    parser.add_argument("--save_dir", type=str, required=True, help="Path to the folder to save the split frames.")
    parser.add_argument("--start_frame", type=int, required=True, help="First frame to split off.")
    parser.add_argument("--end_frame", type=int, required=True, help="Last frame to split off.")
    parser.add_argument("--deconv", type=str, choices=["deconv_family_machine", "deconv_well", "no_deconv"],
                        default="no_deconv", help="Deconvolution type that should be performed, defaults to no deconv.")
    parser.add_argument("--loglevel", type=int, default=7, help="Loglevel of the script.")
    args = parser.parse_args()

    # run the main
    main(**vars(args))


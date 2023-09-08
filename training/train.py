import re
from configparser import ConfigParser
from pathlib import Path

import click
import datetime
import tensorflow as tf

from midap.data.tf_pipeline import TFPipe
from midap.networks.evaluation import tf_metrics
from midap.utils import get_logger


def configure(ctx: click.Context, param: click.Option, filename: str):
    """
    The callback for the config file parameter that sets the default map
    :param ctx: The context
    :param param: The name of the parameter
    :param filename: The file name of the config file
    """
    cfg = ConfigParser()
    cfg.read(filename)
    try:
        options = dict(cfg['TFTraining'])
    except KeyError:
        options = {}
    ctx.default_map = options


def get_files(ctx: click.Context, param: click.Argument, filename: tuple):
    """
    The callback for the file_name arguments of the main function. Arguments won't be mapped through the default map
    defined in the configure function, so we need to manually extract them if provided.
    :param ctx: The context
    :param param: The name of the parameter
    :param filename: The file name of the train file(s)
    """

    # if we just want to gen a config file we don't care
    if "gen_config" in ctx.params:
        return filename

    # if we have something specified we just return it
    if len(filename) > 0:
        return list(filename)
    else:
        # check if its in the default map
        filename = ctx.default_map.get(param.name, None)
        if filename is None:
            raise click.BadParameter("Please specify at least one filename as argument or in config file")
        # return the extraced filenames
        filename = re.findall('\"(.*)\"', filename)
        if len(filename) == 0:
            raise click.BadParameter('No valid filename found in the config file, please make sure each file name '
                                     'is surrounded by quites, e.g.  "img_1_raw.tif"')
        return filename


@click.command(context_settings={"ignore_unknown_options": True})
@click.option('-c', '--config', type=click.Path(dir_okay=False), default='tf_train_config.ini', callback=configure,
              is_eager=True, expose_value=False, help='Read option defaults from the specified INI file',
              show_default=True, )
# general
@click.option('--gen_config', type=click.Path(), default=None,
              help="Creates a config file with all fields set and exits.")
@click.option('--loglevel', type=int, default=7,
              help='Loglevel of the script can range from 0 (no output) to 7 (debug, default)')
# training data
@click.option('--n_grid', type=int, default=4,
              help='The grid used to split the original image into distinct patches for train, test and val dsets')
@click.option('--test_size', type=float, default=0.15, help='Ratio for the test set')
@click.option('--val_size', type=float, default=0.2, help='Ratio for the validation set')
@click.option('--sigma', type=float, default=2.0, help='sigma parameter used for the weight map calculation [1]')
@click.option('--w_0', type=float, default=2.0, help='w_0 parameter used for the weight map calculation [1]')
@click.option('--w_c0', type=float, default=1.0,
              help='basic class weight for non-cell pixel parameter used for the weight map calculation [1]')
@click.option('--w_c1', type=float, default=1.1,
              help='basic class weight for cell pixel parameter used for the weight map calculation [1]')
@click.option('--loglevel', type=int, default=7,
              help='The loglevel of the logger instance, 0 -> no output, 7 (default) -> max output')
@click.option('--np_random_seed', type=int, default=11, help="The numpy random seed that should be set.")
@click.option('--batch_size', type=int, default=32, help='The batch size of the data sets for training.')
@click.option('--shuffle_buffer', type=int, default=128, help='The shuffle buffer used for the training set')
@click.option('--image_size', type=(int, int, int), default=(128, 128, 1),
              help='The target image size including channel dimension')
@click.option('--delta_gamma', type=float, default=0.1,
              help='The max delta_gamma for random gamma adjustments, can be None -> no adjustments')
@click.option('--delta_gain', type=float, default=0.1,
              help='The max delta_gain for random gamma adjustments, can be None -> no adjustments')
@click.option('--delta_brightness', type=float, default=0.4,
              help='The max delta_brightness for random brightness adjustments, can be None -> no adjustments')
@click.option('--delta_brightness', type=float, default=0.4,
              help='The max delta_brightness for random brightness adjustments, can be None -> no adjustments')
@click.option('--lower_contrast', type=float, default=0.2,
              help='The lower limit for random contrast adjustments, can be None -> no adjustments')
@click.option('--upper_contrast', type=float, default=0.5,
              help='The upper limit for random contrast adjustments, can be None -> no adjustments')
@click.option('--rescale', is_flag=True,
              help="If set, all images are rescaled between 0 and 1, note this will undo the contrast and brightness "
                   "adjustments.")
@click.option('--n_repeats', type=int, default=50,
              help='The number of repeats of random operations per original image, i.e. number of data augmentations')
@click.option('--train_seed', type=(int, int), default=None,
              help='A tuple of two seed used to seed the stateless random operations of the training dataset. '
                   'If set to None (default) each iteration through the training set will have different random '
                   'augmentations, if set the same augmentations will be used every iteration. Note that even if this '
                   'seed is set, the shuffling operation will still be truly random if the shuffle_buffer > 1')
@click.option('--val_seed', type=(int, int), default=(11, 12),
              help='The seed for the validation set (see train_seed), defaults to (11, 12) for reproducibility')
@click.option('--test_seed', type=(int, int), default=(13, 14),
              help='The seed for the test set (see train_seed), defaults to (13, 14) for reproducibility')
# train specific params
@click.option('--batch_size', type=int, default=2, help='Batch size used for the training.')
@click.option('--epochs', type=int, default=50, help='Number of epochs used for the training.')
@click.option('--iou_threshold', type=float, default=0.9,
              help='IoU threshold for the AveragePrecision (average Jaccard index for a given threshold) metric.')
@click.option('--custom_model', type=str, default=None,
              help='Name of the class of the custom model to train, this class has to be implemented in '
                   'custom_model.py and has to accept input_size, dropout and metrics as keyword arguments in the '
                   'constructor method.')
@click.option('--restore_path', type=click.Path(), default=None,
              help='Path to restore the model from, note that it will use the model.save_weights routine')
@click.option('--tfboard_logdir', type=click.Path(), default=None,
              help='Logdir used for the Tensorboard callback, defaults to None -> no callback.')
@click.option("--save_path", type=click.Path(exists=True, file_okay=False), default=".",
              help="The path in which the results should be saved, this directory should exists.")
@click.option('--save_model', is_flag=True,
              help='If this flag is set, the model will be saved using tf.keras.models.save_model instead of just '
                   'saving the weights.')
@click.argument('train_files', nargs=-1, type=click.Path(), callback=get_files)
def main(**kwargs):
    """
    A function that performs model training
    """
    # create the config
    config = ConfigParser()
    config.read_dict({"TFTraining": {k: f"{v}" for k, v in kwargs.items()}})
    # remove the config name
    config.remove_option("TFTraining", "gen_config")

    # format to the train_files, this can be None in case we just want a config
    if len(kwargs["train_files"]) == 0:
        train_files = ['This is a filename', 'Please do not forget the quotes']
    else:
        train_files = kwargs["train_files"]
    config.set("TFTraining", "train_files", " \n".join([f'\"{f}\"' for f in train_files]))

    # to file
    if kwargs["gen_config"] is not None:
        with open(kwargs["gen_config"], "w+") as f:
            config.write(f)
        return

    # logging
    logger = get_logger(__file__, kwargs["loglevel"])

    # save the config
    config_file = Path(kwargs["save_path"]).joinpath("train_config.ini")
    logger.info(f'Saving config to: {config_file}')
    with open(config_file, "w+") as f:
        config.write(f)

    print(kwargs["train_files"])

    # create the TFPipe
    logger.info("Initializing the data pipelines...")
    tf_pipe = TFPipe(paths=kwargs["train_files"], n_grid=kwargs["n_grid"], test_size=kwargs["test_size"],
                     val_size=kwargs["val_size"], sigma=kwargs["sigma"], w_0=kwargs["w_0"], w_c0=kwargs["w_c0"],
                     w_c1=kwargs["w_c1"], loglevel=kwargs["loglevel"], np_random_seed=kwargs["np_random_seed"],
                     batch_size=kwargs["batch_size"], shuffle_buffer=kwargs["shuffle_buffer"],
                     image_size=kwargs["image_size"], delta_gamma=kwargs["delta_gamma"],
                     delta_gain=kwargs["delta_gain"], delta_brightness=kwargs["delta_brightness"],
                     lower_contrast=kwargs["lower_contrast"], upper_contrast=kwargs["upper_contrast"],
                     rescale=kwargs["rescale"], n_repeats=kwargs["n_repeats"], train_seed=kwargs["train_seed"],
                     val_seed=kwargs["val_seed"], test_seed=kwargs["val_seed"])

    # import the right model
    if kwargs["custom_model"] is None:
        logger.info("Loading standard UNet")
        from midap.networks.unets import UNetv1 as ModelClass
    else:
        logger.info(f'Loading custom class {kwargs["custom_model"]}')
        import custom_model
        ModelClass = getattr(custom_model, kwargs["custom_model"])

    # initialize the model
    model = ModelClass(input_size=kwargs["image_size"], dropout=0.5,
                       metrics=[tf_metrics.AveragePrecision(kwargs['iou_threshold']),
                                tf_metrics.ROIAccuracy()])

    # load the weights
    if (restore_path := kwargs["restore_path"]) is not None:
        logger.info(f"Restoring weights from: {restore_path}")
        model.load_weights(restore_path)

    # callbacks
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5),
                 tf_metrics.ToggleMetrics(toggle_metrics=["average_precision"])]
    if (log_dir := kwargs["tfboard_logdir"]) is not None:
        logger.info(f"Setting TF board log dir: {log_dir}")
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir))

    # Fit the model
    model.fit(x=tf_pipe.dset_train,
              epochs=kwargs["epochs"],
              validation_data=tf_pipe.dset_val,
              callbacks=callbacks)

    # save the results
    if kwargs["save_model"]:
        logger.info(f'Saving model to: {kwargs["save_path"]}')
        model.save(Path(kwargs["save_path"]).joinpath("model.h5"))
    else:
        logger.info(f'Saving weights to: {kwargs["save_path"]}')
        weights_filename = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        model.save_weights(Path(kwargs["save_path"]).joinpath(weights_filename+".h5"), save_format="h5")


if __name__ == '__main__':
    main()

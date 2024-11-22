import re
from configparser import ConfigParser
from pathlib import Path

import click
import datetime
import tensorflow as tf

from midap.data.tf_pipeline import TFPipeMotherMachine
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
        options = dict(cfg["TFTraining"])
        # convert to the right types
        for k, v in options.items():
            if v == "None":
                options[k] = None
            if k == "image_size":
                options[k] = tuple(map(int, re.findall(r"\d+", options[k])))

    except KeyError:
        options = {}
    ctx.default_map = options


@click.command(context_settings={"ignore_unknown_options": True})
@click.option(
    "-c",
    "--config",
    type=click.Path(dir_okay=False),
    default="tf_train_config.ini",
    callback=configure,
    is_eager=True,
    expose_value=False,
    help="Read option defaults from the specified INI file",
    show_default=True,
)
# the data paths
@click.option(
    "--img_dir",
    type=click.Path(exists=True, file_okay=False),
    default="./img",
    help="The path to the directory containing the images.",
)
@click.option(
    "--seg_dir",
    type=click.Path(exists=True, file_okay=False),
    default="./seg",
    help="The path to the directory containing the segmentations.",
)
@click.option(
    "--wei_dir",
    type=click.Path(exists=True, file_okay=False),
    default="./wei",
    help="The path to the directory containing weight maps.",
)
# general
@click.option(
    "--gen_config",
    type=click.Path(),
    default=None,
    help="Creates a config file with all fields set and exits.",
)
@click.option(
    "--loglevel",
    type=int,
    default=7,
    help="Loglevel of the script can range from 0 (no output) to 7 (debug, default)",
)
# training data
@click.option("--test_size", type=float, default=0.15, help="Ratio for the test set")
@click.option(
    "--val_size", type=float, default=0.2, help="Ratio for the validation set"
)
@click.option(
    "--loglevel",
    type=int,
    default=7,
    help="The loglevel of the logger instance, 0 -> no output, 7 (default) -> max output",
)
@click.option(
    "--np_random_seed",
    type=int,
    default=11,
    help="The numpy random seed that should be set.",
)
@click.option(
    "--shuffle_buffer",
    type=int,
    default=128,
    help="The shuffle buffer used for the training set",
)
@click.option(
    "--image_size",
    type=(int, int, int),
    default=(256, 32, 1),
    help="The target image size including channel dimension",
)
# train specific params
@click.option(
    "--batch_size", type=int, default=32, help="Batch size used for the training."
)
@click.option(
    "--epochs", type=int, default=5, help="Number of epochs used for the training."
)
@click.option(
    "--iou_threshold",
    type=float,
    default=0.9,
    help="IoU threshold for the AveragePrecision (average Jaccard index for a given threshold) metric.",
)
@click.option(
    "--custom_model",
    type=str,
    default=None,
    help="Name of the class of the custom model to train, this class has to be implemented in "
    "custom_model.py and has to accept input_size, dropout and metrics as keyword arguments in the "
    "constructor method.",
)
@click.option(
    "--restore_path",
    type=click.Path(),
    default=None,
    help="Path to restore the model from, note that it will use the model.save_weights routine",
)
@click.option(
    "--tfboard_logdir",
    type=click.Path(),
    default=None,
    help="Logdir used for the Tensorboard callback, defaults to None -> no callback.",
)
@click.option(
    "--save_path",
    type=click.Path(exists=True, file_okay=False),
    default=".",
    help="The path in which the results should be saved, this directory should exists.",
)
@click.option(
    "--save_model",
    is_flag=True,
    help="If this flag is set, the model will be saved using tf.keras.models.save_model instead of just "
    "saving the weights.",
)
def main(**kwargs):
    """
    A function that performs model training
    """
    # create the config
    config = ConfigParser()
    config.read_dict({"TFTraining": {k: f"{v}" for k, v in kwargs.items()}})
    # remove the config name
    config.remove_option("TFTraining", "gen_config")

    # to file
    if kwargs["gen_config"] is not None:
        with open(kwargs["gen_config"], "w+") as f:
            config.write(f)
        return

    # logging
    logger = get_logger(__file__, kwargs["loglevel"])

    # save the config
    config_file = Path(kwargs["save_path"]).joinpath("train_config.ini")
    logger.info(f"Saving config to: {config_file}")
    with open(config_file, "w+") as f:
        config.write(f)

    # create the TFPipe
    logger.info("Initializing the data pipelines...")
    tf_pipe = TFPipeMotherMachine(
        img_dir=kwargs["img_dir"],
        seg_dir=kwargs["seg_dir"],
        weight_dir=kwargs["wei_dir"],
        test_size=kwargs["test_size"],
        val_size=kwargs["val_size"],
        loglevel=kwargs["loglevel"],
        np_random_seed=kwargs["np_random_seed"],
        batch_size=kwargs["batch_size"],
        shuffle_buffer=kwargs["shuffle_buffer"],
        image_size=kwargs["image_size"],
    )

    # import the right model
    if kwargs["custom_model"] is None:
        logger.info("Loading standard UNet")
        from midap.networks.unets import UNetv1 as ModelClass
    else:
        logger.info(f'Loading custom class {kwargs["custom_model"]}')
        import custom_model

        ModelClass = getattr(custom_model, kwargs["custom_model"])

    # initialize the model
    model = ModelClass(
        input_size=kwargs["image_size"],
        dropout=0.5,
        metrics=[
            tf_metrics.AveragePrecision(kwargs["iou_threshold"]),
            tf_metrics.ROIAccuracy(),
        ],
    )

    # load the weights
    if (restore_path := kwargs["restore_path"]) is not None:
        logger.info(f"Restoring weights from: {restore_path}")
        model.load_weights(restore_path)

    # callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5),
        tf_metrics.ToggleMetrics(toggle_metrics=["average_precision"]),
    ]
    if (log_dir := kwargs["tfboard_logdir"]) is not None:
        logger.info(f"Setting TF board log dir: {log_dir}")
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir))

    # Fit the model
    model.fit(
        x=tf_pipe.dset_train,
        epochs=kwargs["epochs"],
        validation_data=tf_pipe.dset_val,
        callbacks=callbacks,
    )

    # save the results
    if kwargs["save_model"]:
        logger.info(f'Saving model to: {kwargs["save_path"]}')
        model.save(Path(kwargs["save_path"]).joinpath("model.h5"))
    else:
        logger.info(f'Saving weights to: {kwargs["save_path"]}')
        weights_filename = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        model.save_weights(
            Path(kwargs["save_path"]).joinpath(weights_filename + ".h5"),
            save_format="h5",
        )


if __name__ == "__main__":
    main()

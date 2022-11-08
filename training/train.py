import tensorflow as tf
import numpy as np
import argparse

from midap.utils import get_logger

# Parsing
#########

parser = argparse.ArgumentParser()
parser.add_argument("--save_path", type=str, required=True,
                    help="The path in which the results should be saved, this directory should exists.")
parser.add_argument("--train_data", type=str, default="./training_data/ZF270g/train/training_data_ZF270g_1.npz",
                    help="Path to the numpy file archive (.npz) that contains the training data")
parser.add_argument("--batch_size", type=int, default=2,
                    help="Batch size used for the training.")
parser.add_argument("--epochs", type=int, default=50,
                    help="Number of epochs used for the training.")
parser.add_argument("--custom_model", type=str, default=None,
                    help="Name of the class of the custom model to train, this class has to be implemented in "
                         "custom_model.py and has to accept input_size and dropout as keyword arguments in "
                         "the constructor method.")
parser.add_argument("--restore_path", type=str, default=None,
                    help="Path to restore the model from, note that it will use the model.save_weights routine")
parser.add_argument("--save_model", action="store_true",
                    help="If this flag is set, the model will be saved using tf.keras.models.save_model "
                         "instead of just saving the weights.")
parser.add_argument("--loglevel", type=int, default=7,
                    help="Loglevel of the script can range from 0 (no output) to 7 (debug, default)")
args = parser.parse_args()

# logging
logger = get_logger(__file__, args.loglevel)

# Load and readout data
logger.info("Loading data...")
data = np.load(args.train_data)

X_train = data['X_train'][:10]
y_train = data['y_train'][:10]
weight_maps_train = data['weight_maps_train'][:10]
ratio_cell_train = data['ratio_cell_train'][:10]
X_val = data['X_val'][:10]
y_val = data['y_val'][:10]
weight_maps_val = data['weight_maps_val'][:10]
ratio_cell_val = data['ratio_cell_val'][:10]
logger.info("done!")

# import the right model
if args.custom_model is None:
    logger.info("Loading standard UNet")
    from midap.networks.unets import UNetv1 as ModelClass
else:
    logger.info(f"Loading custom class {args.custom_model}")
    import custom_model
    ModelClass = getattr(custom_model, args.custom_model)

# initialize the model
model = ModelClass(input_size=X_train.shape[1:], dropout=0.5)

# load the weights
if args.restore_path is not None:
    logger.info(f"Restoring weights from: {args.restore_path}")
    model.load_weights(args.restore_path)

# Fit the model
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
model.fit(x=[X_train,
             weight_maps_train,
             y_train], # we need to provide this here for the custom loss
          y=y_train, # here for the accuracy metric
          sample_weight=ratio_cell_train,
          epochs=args.epochs,
          validation_data=([X_val,
                            weight_maps_val,
                            y_val],
                            y_val),
          batch_size=args.batch_size,
          callbacks=[callback],
          shuffle=True)

# save the results
if args.save_model:
    logger.info(f"Saving model to: {args.save_path}")
    model.save(args.save_path)
else:
    logger.info(f"Saving weights to: {args.save_path}")
    model.save_weights(args.save_path)

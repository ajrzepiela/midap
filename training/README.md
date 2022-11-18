# Training models with MIDAP

This directory concerns the customization of the segmentation models used in the MIDAP pipeline. You can:

- Pre-process new training data.
- Visualize the preprocessed data.
- Train standard models from scratch.
- Finetune the parameters from standard UNets.
- Create and train your own models.

## Preparation

To download example data that can be used to run the scripts and notebooks run
```
./download_data.sh
```

## Data Pre-processing

To train or finetune models you need at least one original image and the corresponding segmentation (labels). After 
downloading the example data, you can find an example of such an original image in 
`training_data/ZF270g/orig_img/img/pos33-eGFP_dual-A10200_cut.png` and an example of the segmentation in 
`training_data/ZF270g/orig_img/mask/pos33-eGFP_dual-A10200_seg_2.png`. The pre-processing set is intended to prepare 
this data for the training. This includes the split into a set for training, validation and testing, as well as, 
patch generation and data augmentation. The pre-processing can be done with the scrip `preprocessing.py`. This script 
accepts a variety of commandline arguments, to see the full signature, simply run `python preprocessing.py --help`. 
Additionally, we provide a simple example in the jupyter notebook [preprocess.ipynb](./notebook/preprocess.ipynb). 
After preprocessing the data, you can visualize the generated training data in the 
[traindata_visualization.ipynb](./notebook/traindata_visualization.ipynb) notebook. 

**Note:** The preprocessing script can be run locally and takes usually less than 10 minutes. 

## Model training and testing

The training of models can be done with the `train.py` script. Again, this script takes various command line arguments 
and the signature can be displayed with `python train.py --help`. Additionally, we provide the notebook 
[training.ipynb](./notebook/training.ipynb) that includes more details about the model training. Using this script 
you can either train the standard UNet from scratch, finetune weights that are already in the pipeline or train custom 
models. After the model is trained, you can perform some tests with the notebook 
[test_models.ipynb](./notebook/test_models.ipynb).

**Note:** Training models can be computationally intense. Using the example data, a single epoch on Euler with 
4CPUs (no GPU) and 8GB RAM takes ~1.5 hours, leading to a total training time of more than one day. You can drastically 
decrease this time by using a GPU or increasing the number of CPU cores.

### Implementing custom models

The `custom_model.py` script includes and example of a possible customized model. The simplest way to create a new 
model is to copy this example and just replace the layer in the `__init__` method. However, you can also change the 
optimizer, etc. Just make sure that you do not change the trace of the inputs as well as the `save` method to stay 
compatible with the rest of the pipeline.

## Adding models to MIDAP

Once you have trained your own models, you can add them to the MIDAP pipeline. In case that you trained or finetuned a 
standard UNet you can simply add the weights to the `../model_weights/model_weights_family_mother_machine/` directory 
and the model will be displayed in the selection of models for the default segmentation method. If you trained a 
custom model, the easiest way to add it to the pipeline is if you saved it as a model, i.e. setting the `--save_model` 
flag in the `train.py` script. Then you can create a new segmentation subclass following the description in the 
main [README.md](../README.md#cell-segmentation). In your new class set 
```
self.segmentation_method = tf.keras.models.load_model(<path_to_model>)
```
where `<path_to_model>` is the path where you saved the model.

## TODO:
    - Data augmentation with `tf.data` API
    - Training of `cellpose` models
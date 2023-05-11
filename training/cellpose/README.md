### Cellpose 

This directory contains the code and notebooks to finetune cellpose models. 
It is meant for development and not for production. You can find the successfully trained models 
in midap.

Note that it is not possible to fine tune Omnipose models directly, but one has to retraint the base 
models and then finetune them accordingly.

#### Omnipose Data

The data can be downloaded from [here](https://osf.io/xmury/).

#### Base Training

The base training was done with the following command:

```bash
python -m omnipose --train --use_gpu --look_one_level_down --dir ${DSET_DIR} --mask_filter _masks --n_epochs 4000 --learning_rate 0.1 --diameter 0 --batch_size 16  --RAdam
```

where `${DSET_DIR}` is the path to the dataset, e.g. `bact_fluor/` for the florescence dataset. This will take 
approximately a week to train, but training for fewer epochs will also yield good results. The logs will tell you where 
the models were saved (a subdirectory of `${DSET_DIR}`).

#### Finetuning

The notebook `transform_data.ipynb` transforms the standard MIDAP training data into the format that is required for
finetuning. Omnipose need labled images and NOT binary images. It is therefore much more sensitive to fused cells. So
make sure that you insepect all images properly before you use them for training.

The finetuning was done with the following command:

```bash
python -m omnipose --train --use_gpu --dir ${DSET_DIR} --mask_filter _seg --n_epochs 50 --pretrained_model ${MODEL_PATH} --learning_rate 0.1 --diameter 0 --batch_size 16  --RAdam
```

where `${DSET_DIR}` is the path to the dataset, e.g. `bact_fluor/` for the florescence dataset and 
`${MODEL_PATH}` is the path to the model that was trained in the previous step. This will take
only a few hours to train. Training for more than 50 epochs will lead to severe overfitting.
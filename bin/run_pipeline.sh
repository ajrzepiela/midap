#!/bin/bash
module load new eth_proxy gcc/4.8.2 python/3.7.1 matlab/R2017b

eval "$(conda shell.bash hook)"
conda activate /cluster/work/sis/cdss/oschmanf/miniconda3/envs/workflow

#jupyter nbconvert --to notebook --execute ../notebooks/Test_widget.ipynb

# -1) Define parameters
PATH_FOLDER="/cluster/work/sis/cdss/oschmanf/ackermann-bacteria-segmentation/data/data_Glen/3_ZF270g-FS144r/_tiff/"
INP_1="GD_04092019AlgCo_Cas_Pos57_PH_Exp001.tiff"
INP_2="GD_04092019AlgCo_Cas_Pos57_GFP_Exp001.tiff"
INP_3="GD_04092019AlgCo_Cas_Pos57_TXRED_Exp001.tiff"

CHANNEL_1="/PH/"
CHANNEL_2="/GFP/"
CHANNEL_3="/TXRED/"

# 0) Convert files
#python convert_files.py --directory ../data/data_Glen/3_ZF270g-FS144r/

# 1) Generate folder structure
echo "generate folder structure"
# make directory if it doesn't exist
make_dir() {
    [ ! -d $1 ] && mkdir -p $1
}

POS=$(echo $INP_1 | grep -Po "[Pp][Oo][Ss][0-9]+")
RAW_IM="raw_im/"
SEG_PATH="xy1/"
CUT_PATH="phase/"
SEG_IM_PATH="seg_im/"

make_dir $PATH_FOLDER$POS
make_dir $PATH_FOLDER$POS$CHANNEL_1
make_dir $PATH_FOLDER$POS$CHANNEL_2
make_dir $PATH_FOLDER$POS$CHANNEL_3

make_dir $PATH_FOLDER$POS$CHANNEL_1$RAW_IM
make_dir $PATH_FOLDER$POS$CHANNEL_2$RAW_IM
make_dir $PATH_FOLDER$POS$CHANNEL_3$RAW_IM

make_dir $PATH_FOLDER$POS$CHANNEL_1$SEG_PATH
make_dir $PATH_FOLDER$POS$CHANNEL_2$SEG_PATH
make_dir $PATH_FOLDER$POS$CHANNEL_3$SEG_PATH

make_dir $PATH_FOLDER$POS$CHANNEL_1$SEG_PATH$CUT_PATH
make_dir $PATH_FOLDER$POS$CHANNEL_2$SEG_PATH$CUT_PATH
make_dir $PATH_FOLDER$POS$CHANNEL_3$SEG_PATH$CUT_PATH

make_dir $PATH_FOLDER$POS$CHANNEL_1$SEG_IM_PATH
make_dir $PATH_FOLDER$POS$CHANNEL_2$SEG_IM_PATH
make_dir $PATH_FOLDER$POS$CHANNEL_3$SEG_IM_PATH

# 2) Split frames
echo "split frames"
#python stack2frames.py --path $PATH_FOLDER$INP_1 --pos $POS --channel $CHANNEL_1
#python stack2frames.py --path $PATH_FOLDER$INP_2 --pos $POS --channel $CHANNEL_2
#python stack2frames.py --path $PATH_FOLDER$INP_3 --pos $POS --channel $CHANNEL_3

# 3) Cut chambers
echo "cut chambers"
echo $PATH_FOLDER$POS$CHANNEL_1$RAW_IM
#python frames2cuts.py --path_ch0 $PATH_FOLDER$POS$CHANNEL_1$RAW_IM --path_ch1 $PATH_FOLDER$POS$CHANNEL_2$RAW_IM --path_ch2 $PATH_FOLDER$POS$CHANNEL_3$RAW_IM
#python frames2cuts.py --path_PH $PATH_FOLDER$POS$CHANNEL_2$RAW_IM
#python frames2cuts.py --path_PH $PATH_FOLDER$POS$CHANNEL_3$RAW_IM

# 4) Segmentation
python main_prediction.py --path_pos $PATH_FOLDER$POS --path_channel $CHANNEL_2 --channel 'eGFP'
python main_prediction.py --path_pos $PATH_FOLDER$POS --path_channel $CHANNEL_3 --channel 'mCherry'

# 5) Conversion


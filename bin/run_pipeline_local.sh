#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate ~/miniconda3/envs/workflow


# -1) Define parameters
python set_parameters.py
source settings.sh


# 0) Convert files
#python convert_files.py --directory ../data/data_Glen/3_ZF270g-FS144r/


# 1) Generate folder structure
echo "generate folder structure"

# make directory if it doesn't exist
make_dir() {
    [ ! -d $1 ] && mkdir -p $1
}

POS=$(echo $INP_1 | grep -Eo "[Pp][Oo][Ss][0-9]+")
RAW_IM="raw_im/"
SEG_PATH="xy1/"
CUT_PATH="phase/"
SEG_IM_PATH="seg_im/"
SEG_MAT_PATH="seg/"

for i in $(seq 1 $NUM_CHANNEL_TYPES); do 
	CH="CHANNEL_$i"
	echo ${!CH}
done

# generate folders for channels
make_dir $PATH_FOLDER$POS
for i in $(seq 1 $NUM_CHANNEL_TYPES); do
        CH="CHANNEL_$i"
        make_dir $PATH_FOLDER$POS${!CH}
done

# generate folders raw_im
for i in $(seq 1 $NUM_CHANNEL_TYPES); do
        CH="CHANNEL_$i"
        make_dir $PATH_FOLDER$POS${!CH}$RAW_IM
done

# generate folders for tracking results
for i in $(seq 1 $NUM_CHANNEL_TYPES); do
        CH="CHANNEL_$i"
        make_dir $PATH_FOLDER$POS${!CH}$SEG_PATH
done

# generate folders for cutout images
for i in $(seq 1 $NUM_CHANNEL_TYPES); do
        CH="CHANNEL_$i"
        make_dir $PATH_FOLDER$POS${!CH}$SEG_PATH$CUT_PATH
done

# generate folders for segmentation images
for i in $(seq 1 $NUM_CHANNEL_TYPES); do
        CH="CHANNEL_$i"
        make_dir $PATH_FOLDER$POS${!CH}$SEG_IM_PATH
done

# generate folders for segmentation-mat files
for i in $(seq 1 $NUM_CHANNEL_TYPES); do
        CH="CHANNEL_$i"
        make_dir $PATH_FOLDER$POS${!CH}$SEG_PATH$SEG_MAT_PATH
done


# 2) Split frames
echo "split frames"
for i in $(seq 1 $NUM_CHANNEL_TYPES); do
        CH="CHANNEL_$i"
        I="INP_$i"
        python stack2frames.py --path $PATH_FOLDER${!I} --pos $POS --channel ${!CH}
done


# 3) Cut chambers
echo "cut chambers"
if [ -z "$CHANNEL_2" ] || [ -z "$CHANNEL_3" ]
    then
        python frames2cuts.py --path_ch0 $PATH_FOLDER$POS$CHANNEL_1$RAW_IM
        echo $PATH_FOLDER$POS$CHANNEL_1$RAW_IM
    else
        python frames2cuts.py --path_ch0 $PATH_FOLDER$POS$CHANNEL_1$RAW_IM --path_ch1 $PATH_FOLDER$POS$CHANNEL_2$RAW_IM --path_ch2 $PATH_FOLDER$POS$CHANNEL_3$RAW_IM
fi


# 4) Segmentation
echo "segment images"
if [[ $CELL_TYPE_1 == "13B01" ]] || [[ $CELL_TYPE_1 == "Zf270g" ]] || [[ $CELL_TYPE_1 == "1F187" ]]
    then
        python main_prediction.py --path_pos $PATH_FOLDER$POS --path_channel $CHANNEL_2 --channel 'eGFP'
fi

if [[ $CELL_TYPE_1 == "FS144" ]] || [[ $CELL_TYPE_1 == "Zf270g" ]]
    then
        python main_prediction.py --path_pos $PATH_FOLDER$POS --path_channel $CHANNEL_2 --channel 'mCherry'
fi

if [[ $CELL_TYPE_2 == "13B01" ]] || [[ $CELL_TYPE_2 == "Zf270g" ]] || [[ $CELL_TYPE_2 == "1F187" ]]
    then
        python main_prediction.py --path_pos $PATH_FOLDER$POS --path_channel $CHANNEL_3 --channel 'eGFP'
fi

if [[ $CELL_TYPE_2 == "FS144" ]] || [[ $CELL_TYPE_2 == "Zf270g" ]]
    then
        python main_prediction.py --path_pos $PATH_FOLDER$POS --path_channel $CHANNEL_3 --channel 'mCherry'
fi


# if [[ $CHANNEL_2 == "/GFP/" ]]
#     then
#         python main_prediction.py --path_pos $PATH_FOLDER$POS --path_channel $CHANNEL_2 --channel 'eGFP'
# fi
# if [[ $CHANNEL_2 == "/mCherry/" ]]
#     then
#         python main_prediction.py --path_pos $PATH_FOLDER$POS --path_channel $CHANNEL_2 --channel 'mCherry'
# fi
# if [[ $CHANNEL_2 == "/TXRED/" ]]
#     then
#         python main_prediction.py --path_pos $PATH_FOLDER$POS --path_channel $CHANNEL_2 --channel 'mCherry'
# fi
# if [[ $CHANNEL_3 == "/GFP/" ]]
#     then
#         python main_prediction.py --path_pos $PATH_FOLDER$POS --path_channel $CHANNEL_3 --channel 'eGFP'
# fi
# if [[ $CHANNEL_3 == "/mCherry/" ]]
#     then
#         python main_prediction.py --path_pos $PATH_FOLDER$POS --path_channel $CHANNEL_3 --channel 'mCherry'
# fi
# if [[ $CHANNEL_3 == "/TXRED/" ]]
#     then
#         python main_prediction.py --path_pos $PATH_FOLDER$POS --path_channel $CHANNEL_3 --channel 'mCherry'
# fi


# 5) Conversion
echo "run file-conversion"
for i in $(seq 2 $NUM_CHANNEL_TYPES); do
        CH="CHANNEL_$i"
        python seg2mat.py --path_cut $PATH_FOLDER$POS${!CH}$SEG_PATH$CUT_PATH --path_seg $PATH_FOLDER$POS${!CH}$SEG_IM_PATH --path_channel $PATH_FOLDER$POS${!CH}
done


# 6) Tracking
echo "run tracking"
for i in $(seq 2 $NUM_CHANNEL_TYPES); do
        CH="CHANNEL_$i"
        /Applications/MATLAB_R2018b.app/matlab -nodisplay -r "tracking_supersegger('$PATH_FOLDER$POS${!CH}')"
done

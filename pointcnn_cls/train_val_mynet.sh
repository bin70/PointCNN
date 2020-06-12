#!/usr/bin/env bash

gpu=
setting=
ckpt_load=
models_folder="../../models"
train_files="../../data/train_files.txt"
val_files="../../data/test_files.txt"

usage() { echo "train/val pointcnn_cls with -g gpu_id -x setting options -l load_ckpt"; }

gpu_flag=0
setting_flag=0
while getopts g:x:l:h opt; do
  case $opt in
  g)
    gpu_flag=1;
    gpu=$(($OPTARG))
    ;;
  x)
    setting_flag=1;
    setting=${OPTARG}
    ;;
  l)
    ckpt_load=${OPTARG}
    ;;
  h)
    usage; exit;;
  esac
done

shift $((OPTIND-1))

if [ $gpu_flag -eq 0 ]
then
  echo "-g option is not presented!"
  usage; exit;
fi

if [ $setting_flag -eq 0 ]
then
  echo "-x option is not presented!"
  usage; exit;
fi

if [ ! -d "$models_folder" ]
then
  mkdir -p "$models_folder"
fi

echo "Train/Val with setting $setting on GPU $gpu!"
CUDA_VISIBLE_DEVICES=$gpu python3 ../train_val_cls.py --log - --no_timestamp_folder -t $train_files -v $val_files -s $models_folder -m pointcnn_cls -l $ckpt_load -x $setting 2>&1 & #> $models_folder/pointcnn_cls_$setting.txt  2>&1 &

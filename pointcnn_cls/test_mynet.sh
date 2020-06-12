#!/usr/bin/env bash

gpu=
setting=
models_folder="../../models/mynet/"
#train_files="../../data/mynet/train_files.txt"
test_files="../../data/mynet/test_files.txt"

usage() { echo "train/val pointcnn_cls with -g gpu_id -x setting options"; }

gpu_flag=0
setting_flag=0
while getopts g:x:h opt; do
  case $opt in
  g)
    gpu_flag=1;
    gpu=$(($OPTARG))
    ;;
  x)
    setting_flag=1;
    setting=${OPTARG}
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

echo "Test with setting $setting on GPU $gpu!"
CUDA_VISIBLE_DEVICES=$gpu python3 ../test_mynet_cls.py -t $test_files -l "/home/elvin/models/mynet/pointcnn_cls_mynet_2019-07-05-20-00-17_31693/ckpts/iter-528" -s $models_folder -m pointcnn_cls -x $setting > $models_folder/pointcnn_cls_$setting.txt 2>&1 &

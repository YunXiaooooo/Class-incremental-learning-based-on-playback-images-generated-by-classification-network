#! /bin/bash
set -e
source activate base


MISAKANUM="10038"
DATASET="flower"
EPOCH="100"
GPUNUM1='2'
GPUNUM2='3'
INVERSIONNUM=20
INVERSIONNUM=$(($INVERSIONNUM/2))
INCR=20
LS_array=(0  0    0  0   0  40   0  40)
LE_array=(20 0    40 0   40 60   40 80)
BS_array=(20 0    40 0   40 20   40 40)

EXP_NAME="/home/tx704/zyx/Dataset/"$DATASET
ARCH_NAME=$(pwd)"/model/"
# 0
conda deactivate
conda activate misaka_sisters

rm -rf $EXP_NAME/reserve2/*
#python main.py --step=0 --misakaNum=$MISAKANUM --datasetName=$DATASET --incr=$INCR --epochs=30
python main.py --step=-1 --misakaNum=$MISAKANUM --datasetName=$DATASET --incr=$INCR --epochs=30



for((step=1;step<5;step++));
do
    sleep 1
    conda deactivate
    conda activate deepInversion
    rm -rf $EXP_NAME/inversion/*

    for((k=0;k<2;k++));
    do
        PY="./../DeepInversion-save/imagenet_inversion.py"
        idx=$(($step-1))
        idx=$(($idx*2))
        idx=$(($idx+$k))
        LS=${LS_array[$idx]}
        LE=${LE_array[$idx]}
        BS=${BS_array[$idx]}
        if [[ $BS -eq "0" ]];
        then
            continue
        fi
        MODELSTEP=$(($step-1))
        if [[ $MODELSTEP -eq "0" ]];
        then
            MODEL=$ARCH_NAME"10032-"$MODELSTEP".pth"
        else
            MODEL=$ARCH_NAME$MISAKANUM"-"$MODELSTEP".pth"
        fi

        for((i=0;i<$INVERSIONNUM;i++))
        do
            python -u -b  $PY --label_start=$LS  --label_end=$LE --bs=$BS --do_flip --exp_name=$EXP_NAME  --arch_name=$MODEL  --store_best_images --gpun=$GPUNUM1 >>inversion_gpu$GPUNUM1.log &
            python -u -b  $PY --label_start=$LS  --label_end=$LE --bs=$BS --do_flip --exp_name=$EXP_NAME  --arch_name=$MODEL  --store_best_images --gpun=$GPUNUM2 | tee inversion_gpu$GPUNUM2.log
            wait
        done
    done
    python copyreserve.py --sourceDir=$EXP_NAME/reserve2/ --targetDir=$EXP_NAME/inversion/
    set +e
    conda deactivate
    conda activate misaka_sisters
    python main.py --step=$step --misakaNum=$MISAKANUM --datasetName=$DATASET --incr=$INCR --epochs=$EPOCH
    while [ $? -ne 0 ]
    do
        python main.py --step=$step --misakaNum=$MISAKANUM --datasetName=$DATASET --incr=$INCR --epochs=$EPOCH
    done
    set -e

done

#!/bin/bash

################
#Load your environments and modules here
################

#HOSTFILE=$(realpath hosts.txt)

cd ../..

export OMP_NUM_THREADS=4
TP_SIZE=2
PP_SIZE=1
BATCH_SIZE=8
WORLD_SIZE=4
MICRO_BATCH_SIZE=$((($BATCH_SIZE / $WORLD_SIZE) / $TP_SIZE))

PROJECT_NAME="colossal_tp${TP_SIZE}_pp${PP_SIZE}_mbs${MICRO_BATCH_SIZE}_gpus${WORLD_SIZE}"
TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)
FULL_PROJECT_NAME="${PROJECT_NAME}-${TIMESTAMP}"
PARENT_TENSORBOARD_DIR="/mnt/vepfs/lcxpt/tensorboard/llama/colossal/"
TENSORBOARD_DIR="${PARENT_TENSORBOARD_DIR}${FULL_PROJECT_NAME}"

CUDA_VISIBLE_DEVICES=3,4,5,6 colossalai run --nproc_per_node $WORLD_SIZE ~/share/ColossalAI/examples/language/llama2/benchmark.py -x -g -b $BATCH_SIZE -p 3d --zero 1 \
  --tp $TP_SIZE --pp $PP_SIZE --num_steps 20 --tensorboard_dir $TENSORBOARD_DIR

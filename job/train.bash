#!/bin/bash
#$-cwd

# load modules 
# (this can be modified based on the ABCI version.)
source /etc/profile.d/modules.sh
module load hpcx/2.12
module load singularitypro/3.11
module load cuda/11.6/11.6.2
module load nccl/2.11/2.11.4-1

# Singularity container path
CONTAINER_PATH="/path/to/your/container.sif"

# job ID is available if you need
JOB_NAME=$JOB_ID

# detect GPU type, V100 or A100
GPU_INFO=$(nvidia-smi --query-gpu=gpu_name --format=csv)
if [[ $GPU_INFO =~ "V100" ]]; then
    NUM_GPUS_PER_NODE=4
elif [[ $GPU_INFO =~ "A100" ]]; then
    NUM_GPUS_PER_NODE=8
else
    readonly PROC_ID=$!
    kill ${PROC_ID}
fi

# get number of GPUs
GPUS_IN_ONE_NODE=$(nvidia-smi --list-gpus | wc -l)
NUM_GPU=$(expr ${NHOSTS} \* ${GPUS_IN_ONE_NODE})
echo "NUM_GPU = ${NUM_GPU}"

# MPI options
MPIOPTS="-np $NUM_GPU -N ${NUM_GPUS_PER_NODE} -x MASTER_ADDR=${HOSTNAME} -hostfile $SGE_JOB_HOSTLIST"

# directories to be mounted
ROOT_SRC="/path/to/your/source/codes/"

# configurations
batch_size=256 # multiple of 16 (2 nodes)
learning_rate=0.0001
amp=fp16

# execute
mpirun $MPIOPTS \
    singularity exec --nv --pwd ${ROOT_SRC}/src/ -B ${ROOT_SRC} \
    ${CONTAINER_PATH} \
    ${ROOT_SRC}/job/python.bash ${ROOT_SRC}/src/train.py \
    --bs ${batch_size} \
    --lr ${learning_rate} \
    --amp ${amp}
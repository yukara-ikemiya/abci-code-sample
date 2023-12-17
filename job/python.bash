#!/bin/bash

# define (copy) environment variables for Huggingface Accelerate
export WORLD_SIZE=$OMPI_COMM_WORLD_SIZE
export RANK=$OMPI_COMM_WORLD_RANK
export LOCAL_RANK=$OMPI_COMM_WORLD_LOCAL_RANK
export MASTER_PORT=11111
exec python "$@"

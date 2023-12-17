# A simple code sample for distributed training on ABCI

This reposity provides an example of training codes for ABCI with support for multi-node training.

## 1. Create a Singularity image file (SIF)

```bash
cd ./docker
./build_docker.bash
./docker2singularity.bash
# simple.sif was created here.
```

## 2. Prepare training codes and job scripts

This repository contains a simple training code with a simple model
with support for multi-node training.

## 3. Run a job on ABCI nodes

You have to specify your group id and compute nodes you want to use.

### Example script:
```bash
qsub -j y -g gce12345 -l rt_AF=1 -l h_rt=0:30:00 ./job/train.bash
```
#!/bin/bash

qsub -j y -g gce12345 -l rt_AF=2 -l h_rt=0:30:00 ./job/train.bash


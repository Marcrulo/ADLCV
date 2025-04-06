#!/bin/bash
#BSUB -J python
#BSUB -q gpua100
#BSUB -W 3:30
#BSUB -gpu "num=1:mode=exclusive_process"

#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2GB]"
#BSUB -n 4 # at least 4 is necessary for the gpua100 queue
##BSUB -u s242723@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o 180_240x12x8_16x100x_1e-4_org_%J.out
#BSUB -e 180_240x12x8_16x100x_1e-4_org_%J.err

# Initialize Python env

. /dtu/3d-imaging-center/courses/conda/conda_init.sh
conda activate adlcv_w1

nvidia-smi
# Load the cuda module
module load cuda/11.6

/appl/cuda/11.6.0/samples/bin/x86_64/linux/release/deviceQuery

python3 ViT_training_loop.py
#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=12:00:00
#SBATCH --job-name=k18
#SBATCH -o k18.o%j
#SBATCH --partition=gpu
#SBATCH --mail-user=jma7@mdanderson.org
#SBATCH --mail-type=begin  
#SBATCH --mail-type=end 

module load gcc/4.9.1 cuda/8.0 cudnn/5.1 python3/3.5.2 tensorflow-gpu/1.0.0

python3 /home/05268/junma7/deepLearning/kaggle18/unet_test.py

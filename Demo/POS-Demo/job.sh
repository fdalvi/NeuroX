#!/bin/bash -l
#SBATCH -J Jupyter_GPU  #name of the job
#SBATCH -o output.txt
#SBATCH -p gpu-all
#SBATCH --gres gpu:1
#SBATCH -c 4
#SBATCH --mem 120000MB
#output file
#queue used
#number of gpus needed
#number of CPUs needed
#amount of RAM needed
module load cuda10.1/toolkit
source activate neuron-analysis
jupyter notebook --ip=*

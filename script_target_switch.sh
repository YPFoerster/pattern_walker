#!/bin/bash -l
#SBATCH --job-name=fpt_histograms_target_switch
#SBATCH --partition=nms_research
#SBATCH --time=0-05:00
#SBATCH --ntasks=8
#SBATCH --output=/scratch/users/%u/patternWalker/%j.out

module load devtools/anaconda
python -u fpt_histograms_sbatch.py $SLURM_NTASKS

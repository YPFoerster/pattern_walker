#!/bin/bash -l
#sbatch --job-name=fpt_histograms
#sbatch --partition=nms_research
#sbatch --time=0-12:00
#sbatch --ntasks=8
#sbatch --output=/scratch/users/%u/patternWalker/%j.out

module load devtools/anaconda
python fpt_histograms_sbatch.py $SLURM_NTASKS

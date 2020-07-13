"""
Adapted from https://vsoch.github.io/lessons/sherlock-jobs/
"""
#!/usr/bin/env python

import os
import subprocess

def mkdir_p(dir):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir):
        os.mkdir(dir)

project_dir='/users/k1801311/patternWalker/test'

mkdir_p(project_dir)
os.chdir(project_dir)
job_directory = "{}/.job".format(os.getcwd())
scratch = '/scratch/user/k1801311'
data_dir = os.path.join(scratch, '/patternWalker/test')

# Make top level directories
mkdir_p(job_directory)
mkdir_p(data_dir)


job_params_dict={'job_name':'A','r':3,'h':4,'gamma':0.02,'N':10,'n_samples':10,
    'n_cores':2}
jobs=[job_params_dict]

for job in jobs:

    job_file = os.path.join(job_directory,"{}.job".format(job["job_name"]))
    job_name_data = os.path.join(data_dir, job["job_name"])

    # Create job_name directories
    mkdir_p(job_name_data)

    with open(job_file) as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines("#SBATCH --job-name={}.job\n".format(job["job_name"]))
        fh.writelines("#SBATCH --output=.out/{}.out\n".format(job["job_name"]))
        fh.writelines("#SBATCH --error=.out/{}.err\n".format(job["job_name"]))
        fh.writelines("#SBATCH --time=0-00:02\n")
        fh.writelines("#SBATCH --mem=1200\n")
        fh.writelines("python test_job.py \
            --branching-factor {r} --height {h} --gamma {gamma} --string_len {N}\
            --num-samples {n_samples} --num-cores {n_cores} --job-id {id} \
            --output-dir {out_dir}\
             \n".format(out_dir=job_name_data,job_id='$SLURM_JOB_ID',**job))

    subprocess.run(os.system("sbatch {}".format(job_file)),shell=True)

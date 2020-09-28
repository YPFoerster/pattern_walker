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

project_dir='{}/simple_histograms'.format(os.getcwd())

mkdir_p(project_dir)
#os.chdir(project_dir)
job_directory = os.path.join(project_dir,".job")
scratch = '/scratch/users/k1801311'
data_dir = os.path.join(scratch, 'patternWalker/simple_histograms')

# Make top level directories
mkdir_p(job_directory)
mkdir_p(data_dir)


job_params_dict={'job_name':'simple_histogram','r':3,'h':4,'gamma':0.3,'N':15,'n_samples':1000,
    'n_cores':2}
jobs=[job_params_dict]

for job in jobs:

    job_file = os.path.join(job_directory,"{}.job".format(job["job_name"]))
    stdout_file =os.path.join(job_directory,"{}.out".format(job["job_name"]))
    err_file=os.path.join(job_directory,"{}.err".format(job["job_name"]))
    job_name_data = os.path.join(data_dir, job["job_name"])

    # Create job_name directories
    mkdir_p(job_name_data)

    with open(job_file,'w') as fh:
        fh.writelines("#!/bin/bash -l\n")
        fh.writelines("#SBATCH --partition=nms_research\n")
        fh.writelines("#SBATCH --job-name={}.job\n".format(job["job_name"]))
        fh.writelines("#SBATCH --output={}\n".format(stdout_file))
        fh.writelines("#SBATCH --error={}\n".format(err_file))
        fh.writelines("#SBATCH --time=0-00:02\n")
        fh.writelines("#SBATCH --mem=1200\n")
        fh.writelines("module load devtools/anaconda\n")
        fh.writelines("python simple_histogram.py \
            --branching-factor {r} --height {h} --gamma {gamma} --string-len {N}\
            --num-samples {n_samples} --num-cores {n_cores} --job-id {id} \
            --job-name {job_name} --output-dir {out_dir}\
             \n".format(out_dir=job_name_data,id='$SLURM_JOB_ID',**job))

    subprocess.run("sbatch {}".format(job_file),shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

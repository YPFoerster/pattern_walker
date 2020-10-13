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


project_name='fractional_overlap_histograms'
project_dir='{}/{}'.format(os.getcwd(),project_name)

mkdir_p(project_dir)
#os.chdir(project_dir)
job_directory = os.path.join(project_dir,".job")
scratch = '/scratch/users/k1801311'
data_dir = os.path.join(scratch, 'patternWalker/{}'.format(project_name))

# Make top level directories
mkdir_p(job_directory)
mkdir_p(data_dir)


#job_params_dicts=[{'job_name':'fpt_histogram_1','r':3,'h':4,'gamma':0.1,'N':15,'n_samples':100000,
#    'n_cores':4}, {'job_name':'fpt_histogram_2','r':3,'h':4,'gamma':0.15,'N':15,'n_samples':100000,
#    'n_cores':4},{'job_name':'fpt_histogram_3','r':3,'h':4,'gamma':0.2,'N':15,'n_samples':100000,
#    'n_cores':4},{'job_name':'fpt_histogram_4','r':3,'h':4,'gamma':0.25,'N':15,'n_samples':100000,
#    'n_cores':4},{'job_name':'fpt_histogram_5','r':3,'h':4,'gamma':0.3,'N':15,'n_samples':100000,
#    'n_cores':4}]

overlap_range=[0.2,0.4,0.6,0.8]
job_params_dicts=[
    {'job_name':'fpt_fractional_overlap_{job_num}'.format(job_num=job_num),'r':2,'seed':0,'gamma':1/6.,'N':15,'overlap':overlap ,'n_samples':100000,
    'n_cores':4} for (job_num,overlap) in zip(range(len(overlap_range)),overlap_range)  
]


jobs=job_params_dicts

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
        fh.writelines("#SBATCH --time=0-02:00\n")
        fh.writelines("#SBATCH --mem=1200\n")
        fh.writelines("module load devtools/anaconda\n")
        fh.writelines("python fractional_overlap_test.py \
            --lam {r} --seed {seed} --gamma {gamma} --string-len {N} --overlap {overlap}\
            --num-samples {n_samples} --num-cores {n_cores} --job-id {id} \
            --job-name {job_name} --output-dir {out_dir}\
             \n".format(out_dir=job_name_data,id='$SLURM_JOB_ID',**job))

    subprocess.run("sbatch {}".format(job_file),shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

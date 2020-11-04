"""
Adapted from https://vsoch.github.io/lessons/sherlock-jobs/
"""
#!/usr/bin/env python

import os
import subprocess
from itertools import product
import numpy as np

def mkdir_p(dir):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir):
        os.mkdir(dir)
        return 1 #return 1 if directory is created because we might want to add it to .gitignore.

project_name='20201104_ptrp_no_seed_combi_o_g'

project_dir='{}/{}'.format(os.getcwd(),project_name)

project_exists=mkdir_p(project_dir)
if not project_exists:
    with open('.gitignore','a') as f:
        f.writelines([project_dir,"\n"])

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


cores=2
#overlap_range=np.arange(0.1,1.1,0.1)#[x*0.05 for x in range(21) ]
#gamma_range=np.arange(0.1,1.1,0.1)#[x*0.05 for x in range(21)]
N=20
param_range=[]
overlap_range=np.arange(0,1.+1/N,1/N)
gamma_range=np.arange(0,1.+1/N,1/N)
param_range=list(product(overlap_range,gamma_range))    

#param_range=list(product(N_range,gamma_range,overlap_range))
n_jobs=len(param_range)
param_range=[(num,*param_range[num]) for num in range(n_jobs) ]
job_params_dicts=[
    {'job_name':'{project_name}_{job_num}'.format(project_name=project_name,job_num=job_num),'r':2,'seed':0,'gamma':gamma,'N':N,'overlap':overlap,'num_refs':20,'n_samples':10000,
    'n_cores':cores} for (job_num,gamma,overlap) in param_range
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
        #fh.writelines("#SBATCH --workdir={}\n".format(project_dir))
        fh.writelines("#SBATCH --output={}\n".format(stdout_file))
        fh.writelines("#SBATCH --error={}\n".format(err_file))
        fh.writelines("#SBATCH --time=0-00:20\n")
        fh.writelines("#SBATCH --mem=200\n")
        fh.writelines("#SBATCH --nodes=1\n")
        fh.writelines("#SBATCH --ntasks={}\n".format(cores))
        fh.writelines("module load devtools/anaconda\n")
        fh.writelines("python ptrp_no_seed.py \
            --lam {r} --gamma {gamma} --string-len {N} --overlap {overlap}\
            --num-samples {n_samples} --num-cores {n_cores} --job-id {id} \
            --job-name {job_name} --output-dir {out_dir}\
             \n".format(out_dir=job_name_data,id='$SLURM_JOB_ID',**job))

    subprocess.run("sbatch {}".format(job_file),shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

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


job_directory = "{}/.job".format(os.getcwd())
scratch = '/scratch/user/k1801311'
data_dir = os.path.join(scratch, '/project')

# Make top level directories
mkdir_p(job_directory)
mkdir_p(data_dir)

job_names=["job_nameA","job_nameB"]

for job_name in job_names:

    job_file = os.path.join(job_directory,"%s.job" %job_name)
    job_name_data = os.path.join(data_dir, job_name)

    # Create job_name directories
    mkdir_p(job_name_data)

    with open(job_file) as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines("#SBATCH --job-name={}.job\n".format(job_name))
        fh.writelines("#SBATCH --output=.out/{}.out\n".format(job_name))
        fh.writelines("#SBATCH --error=.out/{}.err\n".format(job_name))
        fh.writelines("#SBATCH --time=0-02:00\n")
        fh.writelines("#SBATCH --mem=1200\n")
        fh.writelines("python $HOME/project/subproject/run.py {} var1 var2\n".format(job_name_data))

    subprocess.run(os.system("sbatch {}".format(job_file)),shell=True)

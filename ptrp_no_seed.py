"""In this version we reset patterns for every iteration. ptrp='poisson tree redraw patterns' """

import random_walker as rw
import utils
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import datetime
import multiprocessing as mp
import copy
import argparse
import functools
import os
import pickle

parser = argparse.ArgumentParser(description="""
    For a fixed target, on a fixed graph with fixed weigthts, we sample the
    first passage time starting from the root.
    """)

parser.add_argument("--lam", default=0.5, type=float, dest='branching_factor',
    help="Poisson branching factor (default: %(default)s)"
    )
parser.add_argument("--gamma", default=0., type=float,
    help="Bit flipping rate (default: %(default)s)"
    )
parser.add_argument("--string-len", default=10, type=int, dest='string_len',
    help="Len of bit strings assigned to each node (default: %(default)s)"
    )
parser.add_argument("--overlap", default=0, type=float, dest='overlap',
    help="Fractional overlap of adjacent string sections (every sections reaches out to the left over\
          the regular boundary by overlap*string_len bits. (default: %(default)s)")

parser.add_argument("--num-samples", default=10, type=int, dest='number_of_samples',
    help="Number of repeated walks stopping at target. (default: %(default)s)"
    )
parser.add_argument("--num-cores", default=1, type=int, dest='num_cores',
    help="Number of cores requested (default: %(default)s)"
    )
parser.add_argument("--job-id", default="unkown", dest="job_id",
    help="SLURM job ID (default: %(default)s)"
    )

parser.add_argument("--job-name", default="unknown", dest="job_name",
    help="Job name as per submission to sbatch (default: %(default)s)"
    )

parser.add_argument("--output-dir", default=".", dest="out_dir", help="Output files will appear here. (default: %(default)s)")

args=parser.parse_args()

lam=args.branching_factor #offspring number
gamma=args.gamma #mutation rate
#bits in a pattern; must be adapted to ensure uniqueness of patterns
N=args.string_len
overlap=args.overlap
#number of realisations of pattern distributions in this case
number_of_samples=args.number_of_samples
num_cores=args.num_cores
job_id=args.job_id #as assigned by SLURM, for instance
job_name=args.job_name #as submitted to SBATCH
out_dir=args.out_dir #where to dump all that output.
args=vars(args)
args['job_dir']=os.getcwd() #store the location of the script for rerefence

os.chdir(out_dir) #This way, we can simply write files without specified paths.


def search_target(sample_number):
    """Copy walker, get new strings and calculate mfpts."""
    _,_,walker=make_tree(lam,N,gamma,overlap)
    walker.set_weights()
    while walker.x != walker.target_node:
        walker.step()
    return walker.t


def make_tree(lam,N,gamma,overlap):
    G,root=utils.poisson_ditree(lam)
    leaves = utils.leaves(G)
    #r=len(list(G.successors(root)))
    #N_eff=r*N
    walker=rw.sectionedPatternWalker(G.copy(),root,N,gamma,overlap)
    walker.set_weights()
    return G,root,walker

fpts=[]
start_time=datetime.datetime.now()
print(args)
print("Start:",start_time)


#G,root,walker=make_tree(lam,N,gamma,overlap)

#args['sections']=walker.sections
#args['target_node']=walker.target_node
#args['mfpt']=utils.mfpt(walker,[(walker.root,walker.target_node)])
#args['duplicate_patterns']=walker.num_pattern_duplicates()

#Only need to do scheduling if we have more than one core.
if num_cores>1:
    with mp.Pool(num_cores) as p:
        print('Enter multiprocessing.')
        for times in p.map(search_target, range(number_of_samples)):
            fpts.append(times)
        print('Finished multiprocessing.')
else:
    for _ in range(number_of_samples):
        fpts.append(search_target(walker))


fpts = [np.real(x) for x in fpts if x is not None]
end_time=datetime.datetime.now()
print("End:",end_time)
run_time=end_time-start_time
failed_searches=number_of_samples-len(fpts)
print("Runing time:",run_time)
print("Failed Searches:",failed_searches)

#fig,ax=plt.subplots()
#hist,_,_=ax.hist(fpts,bins=50,color='b',alpha=0.7,density=True)
#plt.savefig('{}.png'.format(job_name))
with open('{}.pkl'.format(job_name), 'wb') as f:
    pickle.dump(args, f)
    pickle.dump(fpts,f)

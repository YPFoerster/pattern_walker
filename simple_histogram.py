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

parser.add_argument("--branching-factor", default=3, type=int, dest='branching_factor',
    help="Branching factor of each node (default: %(default)s)"
    )
parser.add_argument("--height", default=3, type=int,
    help="Height of the tree (default: %(default)s)"
    )
parser.add_argument("--gamma", default=0., type=float,
    help="Bit flipping rate (default: %(default)s)"
    )
parser.add_argument("--string-len", default=10, type=int, dest='string_len',
    help="Len of bit strings assigned to each node (default: %(default)s)"
    )
parser.add_argument("--num-samples", default=10, type=int, dest='number_of_samples',
    help="Number of repeated walks stopping at target. (default: %(default)s)"
    )
parser.add_argument("--num-cores", default=1, type=int, dest='num_cores',
    help="Number of cores requested (default: %(default)s)"
    )
parser.add_argument("--job-id", default="unkown", dest="job_id",
    help="SLURM job ID (default: %(default)s)"
    )

parser.add_argument("--job-name", default="unkown", dest="job_name",
    help="Job name as per submission to sbatch (default: %(default)s)"
    )

parser.add_argument("--output-dir", default=".", dest="out_dir", help="Output files will appear here. (default: %(default)s)")

args=parser.parse_args()

r=args.branching_factor #offspring number
h=args.height #height
gamma=args.gamma #mutation rate
#bits in a pattern; must be adapted to ensure uniqueness of patterns
N=args.string_len
#number of realisations of pattern distributions in this case
number_of_samples=args.number_of_samples
num_cores=args.num_cores
job_id=args.job_id #as assigned by SLURM, for instance
job_name=args.job_name #as submitted to SBATCH
out_dir=args.out_dir #where to dump all that output.

os.chdir(out_dir) #This way, we can simply write files without specified paths.


def search_target(walker_instance):
    """Copy walker, get new strings and calculate mfpts."""
    walker=copy.deepcopy(walker_instance)
    while walker.x != walker.target_node:
        walker.step()
    return walker.t

fpts=[]
start_time=datetime.datetime.now()
print("Start:",start_time)
print(vars(args))

G,root=utils.balanced_directed_tree(r,h)
leaves = utils.leaves(G)
walker=rw.patternWalker(G.copy(),root,N,gamma)
walker.set_weights()

facts={'mfpt':utils.mfpt(walker,[(walker.root,walker.target_node)]),
    'duplicate patterns':walker.num_pattern_duplicates()}


#Only need to do scheduling if we have more than one core.
if num_cores>1:
    with mp.Pool(num_cores) as p:
        print('Enter multiprocessing.')
        for times in p.map(search_target, [walker]*number_of_samples):
            fpts.append(times)
        print('Finished multiprocessing.')
else:
    for _ in range(number_of_samples):
        times=search_target(walker)
        for t in times.values():
            fpts.append(t)

fpts = [np.real(x) for x in fpts if x is not None]
end_time=datetime.datetime.now()
print(end_time)
run_time=end_time-start_time
failed_searches=number_of_samples-len(fpts)

fig,ax=plt.subplots()
hist,_,_=ax.hist(fpts,bins=50,color='b',alpha=0.7,density=True)
#plt.text(0.7,0.7,'mean={m},std={s}'.format(m=round(np.mean(fpts),2),s=round(np.std(fpts),2)),transform=ax.transAxes)
#ax.set_title(vars(args))
#plt.savefig(os.path.join(output_loc,'FPT'+name_string+'.pdf'))
#plt.savefig(os.path.join(output_loc,'FPT'+name_string+'.png'))
plt.savefig('{}.png'.format(job_name))
with open('{}.pkl'.format(job_name), 'ab') as f:
    pickle.dump(fpts,f)
    pickle.dump(vars(args), f)
    pickle.dump(facts,f)

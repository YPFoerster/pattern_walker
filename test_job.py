import random_walker as rw
import utils
import networkx as nx
from networkx.generators import balanced_tree
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import sys
import multiprocessing as mp
import copy
import argparse
import functools

parser = argparse.ArgumentParser(description="""
    This script mimics an instance of our mfpts sampling.
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
    help="Number of pattern realisations (default: %(default)s)"
    )
parser.add_argument("--num-cores", default=1, type=int, dest='num_cores',
    help="Number of cores requested (default: %(default)s)"
    )
parser.add_argument("--job-id", default="unkown", dest="job_id",
    help="SLURM job ID (default: %(default)s)"
    )

parser.add_argument("--output-dir", default=".", dest="out_dir", help="Output files will appear here. (default: %(default)s)")

args=parser.parse_args()

r=args.branching_factor #offspring number
h=args.height #height
gamma=args.gamma #mutation rate
N=args.string_len #bits in a pattern; must be adapted to ensure uniqueness of patterns
number_of_samples=args.number_of_samples
num_cores=args.num_cores
job_id=args.job_id
out_dir=args.out_dir




def solve_mfpts(walker_instance,pairs):
    walker=copy.deepcopy(walker_instance)
    walker.reset_patterns()
    walker.set_weights()
    return utils.mfpt(walker,pairs,weight_str='prob')

def unpack_arg_decorator(func):
    @functools.wraps(func)
    def wrapper(arglist,**kwargs):
        return func(*arglist,**kwargs)
    return wrapper

fpts=[]
#name_string='r{r}h{h}gamma{gamma}N{N}'.format(r=r,h=h,gamma=str(round(gamma,gamma_roundoff)).replace('.','-'),N=N,number_of_samples=number_of_samples, max_time=max_time)
start_time=datetime.datetime.now()
print("Start:",start_time)
print(vars(args))
G=balanced_tree(r,h)
root=None
#root is the inly node in G with degree r, all others have degree r+1.
for node in G.nodes():
    if nx.degree(G,node)==r:
        root=node
        break
G=utils.directify(G,root)[0]
leaves = utils.leaves(G)
walker=rw.patternWalker(G.copy(),root,N,gamma)
walker.set_weights()
pairs=[(walker.root,x) for x in leaves]
if num_cores>1:
    solve_mfpts=unpack_arg_decorator(solve_mfpts)
    with mp.Pool(num_cores) as p:
        print('Enter multiprocessing.')
        for times in p.map(solve_mfpts, zip([walker]*number_of_samples,[pairs]*number_of_samples)):
            for t in times.values():
                fpts.append(t)
        print('Finished multiprocessing.')
else:
    for _ in range(number_of_samples):
        times=solve_mfpts(walker,pairs)
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
plt.show()

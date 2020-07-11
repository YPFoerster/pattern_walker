import random_walker as rw
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

parser = argparse.ArgumentParser(description="""
    This script mimics an instance of our mfpts sampling.
    """)

parser.add_argument("--branching-factor", default=3, type=int, dest='r',
    help="Branching factor of each node (default: %(default)s)"
    )
parser.add_argument("--height", default=3, type=int,
    help="Height of the tree (default: %(default)s)"
    )
parser.add_argument("--gamma", default=0., type=float,
    help="Bit flipping rate (default: %(default)s)"
    )
parser.add_argument("--string-len", default=10, type=int, dest='N',
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

args=parser.parse_args()

r=args.r #offspring number
h=args.height #height
gamma=args.gamma #mutation rate
#gamma_roundoff=3 #for display and file names
N=args.N #bits in a pattern; must be adapted to ensure uniqueness of patterns
number_of_samples=args.number_of_samples
#max_time=int(sys.argv[6])
num_cores=args.num_cores
job_id=args.job_id

print(r)
print(h)
"""
def search(walker_instance):
    walker=copy.copy(walker_instance)
    for i in range(max_time):
        walker.step()
        if walker.metric(walker.G.nodes[walker.x]['pattern'],walker.searched_pattern)==0 and walker.x==walker.searched_node:
            return walker.t


fpts=[]
name_string='r{r}h{h}gamma{gamma}N{N}'.format(r=r,h=h,gamma=str(round(gamma,gamma_roundoff)).replace('.','-'),N=N,number_of_samples=number_of_samples, max_time=max_time)
start_time=datetime.datetime.now()
G=balanced_tree(r,h)
root=None
#root is the inly node in G with degree r, all others have degree r+1.
for node in G.nodes():
    if nx.degree(G,node)==r:
        root=node
        break
G=directify(G,root)[0]
target = leaves(G)[0]
walker=rw.patternWalker(G.copy(),root,N,gamma,search_for=target)
walker.set_weights()

with mp.Pool(num_cores) as p:
    print('Enter multiprocessing.')
    for time in p.map(search, [walker]*number_of_samples):
        fpts.append(time)
    print('Finished multiprocessing.')
fpts = [x for x in fpts if x is not None]
end_time=datetime.datetime.now()
run_time=end_time-start_time
failed_searches=number_of_samples-len(fpts)

fig,ax=plt.subplots()
hist,_,_=ax.hist(fpts,bins=50,color='b',alpha=0.7,density=True)
plt.text(0.7,0.7,'mean={m},std={s}'.format(m=round(np.mean(fpts),2),s=round(np.std(fpts),2)),transform=ax.transAxes)
plt.title('r={r}, h={h}, $\Gamma$={gamma}, N={N}, samples={number_of_samples}, max_time={max_time}, fails={fails}'.format(r=r,h=h,gamma=round(gamma,gamma_roundoff),N=N,number_of_samples=number_of_samples, max_time=max_time,fails=failed_searches))
plt.savefig(os.path.join(output_loc,'FPT'+name_string+'.pdf'))
plt.savefig(os.path.join(output_loc,'FPT'+name_string+'.png'))

"""

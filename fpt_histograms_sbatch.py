import random_walker as rw
from dag_exps import poisson_ditree,directify,leaves
import networkx as nx
from networkx.generators import balanced_tree
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import sys
import multiprocessing as mp
import copy

r=3 #offspring number
h=5 #height
gamma=0.2 #mutation rate
N=100 #bits in a pattern; must be adapted to ensure uniqueness of patterns
number_of_samples=50
max_time=500
num_cores=sys.argv[1]

scratch_loc='/scratch/users/k1801311/patternWalker'
log_file=os.path.join(scratch_loc,'log.out')
output_loc=os.path.join(scratch_loc,'outputs/')

def search(walker_instance):
    walker=copy.copy(walker)
    for i in range(max_time):
        walker.step()
        if walker.metric(walker.G.nodes[walker.x]['pattern'],walker.searched_pattern)==0 and walker.x==walker.searched_node:
            return walker.t



with open(log_file,mode='a') as f:
    f.writelines("########################################################################")
for r in [5]:
    for h in [7]:
        for gamma in [0.2,0.5,0.8]:
            name_string='r{r}h{h}gamma{gamma}N{N}'.format(r=r,h=h,gamma=str(round(gamma,2)).replace('.','-'),N=N,number_of_samples=number_of_samples, max_time=max_time)
            start_time=datetime.datetime.now()
            with open(log_file,mode='a') as f:
                f.writelines(start_time.strftime("#%Y-%m-%d %H:%M:%S"))
                f.writelines(name_string)
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
            with open(log_file,mode='a') as f:
                f.writelines('Number of duplicate patterns: '+str(rw.count_pattern_duplicates(walker)))
            walker.set_weights()

            p=mp.Pool(num_cores)
            fpts=p.map(search, [walker]*number_of_samples)
            p.close()
            p.join()

            end_time=datetime.datetime.now()
            run_time=datetime.timedelta(start_time,end_time)
            failed_searches=number_of_samples-len(fpts)
            with open(log_file,mode='a') as f:
                f.writelines('Runtime:', run_time)
                f.writelines('Failed searches: '+str(failed_searches))
            fig,ax=plt.subplots()
            hist,_,_=ax.hist(fpts,bins=50,color='b',alpha=0.7,density=True)
            plt.text(0.7,0.7,'mean={m},std={s}'.format(m=round(np.mean(fpts),2),s=round(np.std(fpts),2)),transform=ax.transAxes)
            plt.title('r={r}, h={h}, $\Gamma$={gamma}, N={N}, samples={number_of_samples}, max_time={max_time}, fails={fails}'.format(r=r,h=h,gamma=round(gamma,2),N=N,number_of_samples=number_of_samples, max_time=max_time,fails=failed_searches))
            plt.savefig(os.path.join(output_loc,'FPT'+name_string+'.pdf'))
            #plt.show()
with open(log_file,mode='a') as f:
    f.write("#########################################################################")

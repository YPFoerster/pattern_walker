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
gamma=0.02 #mutation rate
gamma_roundoff = 2 #just to trim gamma in filenames
N=100 #bits in a pattern; must be adapted to ensure uniqueness of patterns
number_of_samples=5000
max_time=5000
num_cores=1
job_id='unknown'
if len(sys.argv)==1:
    pass
elif len(sys.argv)>1:
    num_cores=int(sys.argv[1])
if len(sys.argv)>2:
    job_id=sys.argv[2]
scratch_loc='/scratch/users/k1801311/patternWalker'
log_file=os.path.join(scratch_loc,'log_target_switch.out')
output_loc=os.path.join(scratch_loc,'outputs/')

def search(walker_instance):
    walker=copy.deepcopy(walker_instance)
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Switching target node here
    #walker.searched_node=np.random.choice(walker.G.nodes())
    #walker.searched_pattern=walker.G.nodes[walker.searched_node]['pattern']
    for i in range(max_time):
        walker.step()
        if walker.metric(walker.G.nodes[walker.x]['pattern'],walker.searched_pattern)==0 and walker.x==walker.searched_node:
            return walker.t



with open(log_file,mode='a') as f:
    f.writelines("########################################################################\n")
    f.write('Job_ID '+job_id+'\n')
for r in [3]:
    for h in [6,8]:
        for gamma in np.arange(0.0,0.12,0.02):
            fpts=[]
            name_string='r{r}h{h}gamma{gamma}N{N}_target_switch'.format(r=r,h=h,gamma=str(round(gamma,gamma_roundoff)).replace('.','-'),N=N,number_of_samples=number_of_samples, max_time=max_time)
            start_time=datetime.datetime.now()
            with open(log_file,mode='a') as f:
                f.writelines([start_time.strftime("#%Y-%m-%d %H:%M:%S"),name_string,'\n'])
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
                f.writelines('Number of duplicate patterns: '+str(rw.count_pattern_duplicates(walker))+'\n')
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
            with open(log_file,mode='a') as f:
                f.write('Runtime:'+ str(run_time)+'\n')
                f.write('Failed searches:' +str(failed_searches)+'\n')
            with open(os.path.join(output_loc,'FPT'+name_string+'.csv'),mode='w') as f:
                f.writelines('{}, '.format(x) for x in fpts)
                f.write('\n')
            fig,ax=plt.subplots()
            hist,_,_=ax.hist(fpts,bins=50,color='b',alpha=0.7,density=True)
            plt.text(0.7,0.7,'mean={m},std={s}'.format(m=round(np.mean(fpts),2),s=round(np.std(fpts),2)),transform=ax.transAxes)
            plt.title('r={r}, h={h}, $\Gamma$={gamma}, N={N}, samples={number_of_samples}, max_time={max_time}, fails={fails}'.format(r=r,h=h,gamma=round(gamma,gamma_roundoff),N=N,number_of_samples=number_of_samples, max_time=max_time,fails=failed_searches))
            plt.savefig(os.path.join(output_loc,'FPT'+name_string+'.pdf'))
            plt.savefig(os.path.join(output_loc,'FPT'+name_string+'.png'))
with open(log_file,mode='a') as f:
    f.write("#########################################################################\n")

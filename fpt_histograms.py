import random_walker as rw
from dag_exps import poisson_ditree,directify,leaves
import networkx as nx
from networkx.generators import balanced_tree
import numpy as np
import matplotlib.pyplot as plt
import datetime

r=3 #offspring number, will be covered by loop down below anyway
h=5 #height, will be covered by looop down below anyway
gamma=0.2 #mutation rate, will be covered by loop down below anyway
N=100 #bits in a pattern; must be adapted to ensure uniqueness of patterns
number_of_samples=500
max_time=500

with open('log.out',mode='a') as f:
    f.write("########################################################################\n")
    for r in [3]:
        for h in [5]:
            for gamma in [0.02]:
                name_string='r{r}h{h}gamma{gamma}N{N}'.format(r=r,h=h,gamma=str(round(gamma,2)).replace('.','-'),N=N,number_of_samples=number_of_samples, max_time=max_time)
                now=datetime.datetime.now()
                f.write(now.strftime("#%Y-%m-%d %H:%M:%S")+'\n')
                f.write(name_string+'\n')
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
                f.write('Number of duplicate patterns: '+str(rw.count_pattern_duplicates(walker))+'\n')
                walker.set_weights()

                fpts=[]

                for iter in range(number_of_samples):
                    for i in range(max_time):
                        print('Iteration ', iter, ', time step ', i,'    ',end='\r')
                        walker.step()
                        if walker.metric(walker.G.nodes[walker.x]['pattern'],walker.searched_pattern)==0 and walker.x==walker.searched_node:
                            fpts.append(walker.t)
                            walker.reset()
                            break
                    #If the inner loop does not find the target, we need to reset separately.
                    #Ensures that we don't get the times of several failed experiments summed up.
                    walker.reset()
                print('\n')


                failed_searches=number_of_samples-len(fpts)
                f.write('Failed searches: '+str(failed_searches)+'\n')
                print('\n')
                fig,ax=plt.subplots()
                hist,_,_=ax.hist(fpts,bins=50,color='b',alpha=0.7,density=True)
                plt.text(0.7,0.7,'mean={m},std={s}'.format(m=round(np.mean(fpts),2),s=round(np.std(fpts),2)),transform=ax.transAxes)
                plt.title('r={r}, h={h}, $\Gamma$={gamma}, N={N}, samples={number_of_samples}, max_time={max_time}, fails={fails}'.format(r=r,h=h,gamma=round(gamma,2),N=N,number_of_samples=number_of_samples, max_time=max_time,fails=failed_searches))
                plt.savefig('./outputs/FPT'+name_string)
                #plt.show()
    f.write("#########################################################################")

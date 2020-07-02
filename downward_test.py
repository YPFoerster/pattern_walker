import random_walker as rw
from dag_exps import poisson_ditree,directify,leaves
import networkx as nx
from networkx.generators import balanced_tree
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.mstats import pearsonr

r=4 #offspring number, will be covered by loop down below anyway
h=5 #height, will be covered by looop down below anyway
N=1000 #bits in a pattern; must be adapted to ensure uniqueness of patterns
gamma=2/N #mutation rate, will be covered by loop down below anyway
number_of_samples=500
max_time=5000
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
print('Number of duplicate patterns: ',str(rw.count_pattern_duplicates(walker)))
walker.set_weights()

#for step in range(max_time):
#    print(' time step ', step,'    ',end='\r')
#    walker.step()
#    if walker.metric(walker.G.nodes[walker.x]['pattern'],walker.searched_pattern)==0 and walker.x==walker.searched_node:
#    #    fpts.append(walker.t)
#    #    walker.reset()
#        break
#print('\n')
profile=walker.mark_downward_steps()
upwardness=np.sum(np.array(profile)>0)

target_distance=[ rw.hamming_dist(walker.searched_pattern,x) for x in nx.get_node_attributes(walker.G,'pattern').values() ]
probs=list(nx.get_edge_attributes(walker.G,'prob').values())
up_flux=np.array([walker.get_upward_flux(site) for site in walker.G])
down_flux=np.array([walker.get_downward_flux(site) for site in walker.G])
net_down_flux=down_flux-up_flux

print(pearsonr(target_distance,net_down_flux))
fig,ax=plt.subplots(3,1)
#ax.plot(up_flux,label='up')
#ax.plot(down_flux,label='down')
#ax.legend(loc='best')
ax[0].hist(net_down_flux,bins=20)
ax[0].set_xlabel('Net down flux')
ax[1].hist(target_distance,bins=20)
ax[1].set_xlabel('Target Hamming distance')
ax[2].hist(probs,bins=20)
#ax.plot(np.cumsum(profile))
#plt.text(0.5,0.5,str(upwardness/len(profile)),transform=ax.transAxes)
plt.tight_layout()
plt.show()

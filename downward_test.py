import random_walker as rw
from dag_exps import poisson_ditree,directify,leaves
import networkx as nx
from networkx.generators import balanced_tree
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.mstats import pearsonr

r=3 #offspring number, will be covered by loop down below anyway
h=5 #height, will be covered by looop down below anyway
N=15 #bits in a pattern; must be adapted to ensure uniqueness of patterns
gamma=5/N #mutation rate, will be covered by loop down below anyway
number_of_samples=500
max_time=5000
G=balanced_tree(r,h)
root=None
name_string='r{r}h{h}gamma{gamma}N{N}'.format(r=r,h=h,N=N,gamma=str(round(gamma,3)).replace('.','-'))
#root is the inly node in G with degree r, all others have degree r+1.
for node in G.nodes():
    if nx.degree(G,node)==r:
        root=node
        break
G=directify(G,root)[0]
target = leaves(G)[0]
walker=rw.patternWalker(G.copy(),root,N,gamma,search_for=target)
print(len(walker.G))
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

fig,ax=plt.subplots(2,2,figsize=(14,8))
ax[0,0].hist(up_flux,label='up',alpha=0.5,bins=20,density=True)
ax[0,0].hist(down_flux,label='down',alpha=0.5,bins=20,density=True)
ax[0,0].legend(loc='best')
ax[0,0].set_xlabel('"Current"')
ax[0,0].set_ylabel('Frequency')
ax[1,0].hist(net_down_flux,bins=50,density=True)
ax[1,0].set_xlabel('Net downward "current"')
ax[0,1].hist(target_distance,bins=20,density=True)
ax[0,1].set_xlabel('Hamming distance from target pattern')
ax[0,1].text(0.2,0.8,'Cor(Net down current,target distance)={}'.format(round(pearsonr(target_distance,net_down_flux)[0],2)), transform=ax[0,1].transAxes)
ax[0,1].set_ylabel('Frequency')
ax[1,1].hist(probs,bins=50,density=True)
ax[1,1].set_xlabel('Transition matrix entry')
ax[1,1].set_ylabel('Frequency')
#ax.plot(np.cumsum(profile))
#plt.text(0.5,0.5,str(upwardness/len(profile)),transform=ax.transAxes)
plt.tight_layout()
plt.savefig('./outputs/downwardness'+name_string+'.pdf')
plt.savefig('./outputs/downwardness'+name_string+'.png')
#plt.show()

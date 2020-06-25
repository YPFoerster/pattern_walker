import random_walker as rw
from dag_exps import poisson_ditree,directify,leaves
import networkx as nx
from networkx.generators import balanced_tree
import numpy as np
import matplotlib.pyplot as plt

r=3 #offspring number
h=5 #height
gamma=0.2 #mutation rate
N=100 #bits in a pattern; must be adapted to ensure uniqueness of patterns
number_of_samples=5000
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
print('Number of duplicate patterns: ', rw.count_pattern_duplicates(walker))
walker.set_weights()

fpts_fixed_pattern=[]
fpts_varying_pattern=[]

print('Gather first set of data.')
for iter in range(number_of_samples):
    for i in range(max_time):
        print('Iteration ', iter, ', time step ', i,'    ',end='\r')
        walker.step()
        if walker.metric(walker.G.nodes[walker.x]['pattern'],walker.searched_pattern)==0 and walker.x==walker.searched_node:
            fpts_fixed_pattern.append(walker.t)
            walker.reset()
            break
    #If the inner loop does not find the target, we need to reset separately.
    #Ensures that we don't get the times of several failed experiments summed up.
    walker.reset()

print('Gather second set of data.')
for iter in range(number_of_samples):
    #print('round ',iter)
    for i in range(max_time):
        print('Iteration ', iter, ', time step ', i,'    ',end='\r')
        walker.step()
        if walker.metric(walker.G.nodes[walker.x]['pattern'],walker.searched_pattern)==0 and walker.x==walker.searched_node:
            fpts_varying_pattern.append(walker.t)
            walker=rw.patternWalker(G.copy(),root,N,gamma,search_for=target)
            walker.set_weights()
            break
    #If the inner loop does not find the target, we need to reset separately.
    #Ensures that we don't get the times of several failed experiments summed up.
    walker.reset()


failed_fixed_pattern=number_of_samples-len(fpts_fixed_pattern)
failed_varying_pattern=number_of_samples-len(fpts_varying_pattern)
print('Failed search with fixed patterns: ',failed_fixed_pattern)
print('Failed search with varying patterns: ',failed_varying_pattern)

fig,ax=plt.subplots()
hist_fixed_pattern,_,_=ax.hist(fpts_fixed_pattern,bins=50,color='b',alpha=0.7,density=True)
hist_varying_pattern,_,_=ax.hist(fpts_varying_pattern,bins=50,color='r',alpha=0.7,density=True)
dist=np.linalg.norm( hist_fixed_pattern-hist_varying_pattern,ord=1)
print('Distance of histograms in L1=',dist)
plt.text(0.5,0.5,'Histogram $L^1$ distance={}'.format( dist ) ,transform=ax.transAxes)
#plt.text(0.7,0.7,'mean={m},std={s}'.format(m=round(np.mean(fpts),2),s=round(np.std(fpts),2)),transform=ax.transAxes)
plt.savefig('fixed_v_varying_patterns.pdf')
plt.show()

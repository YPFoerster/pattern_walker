import random_walker as rw
from dag_exps import poisson_ditree,directify
import networkx as nx
from networkx.generators import balanced_tree
import numpy as np
import matplotlib.pyplot as plt

#np.random.seed(0)
G=balanced_tree(3,5)
root=None
#root is the inly node in G with degree r, all others have degree r+1.
for node in G.nodes():
    if nx.degree(G,node)==3:
        root=node
        break
G=directify(G,root)[0]
walker=rw.patternWalker(G,root,200,0.1)
walker.set_weights()
#np.random.seed()

fpts=[]

for _ in range(2000):
    for i in range(1000):
        walker.step()
        if walker.metric(walker.G.nodes[walker.x]['pattern'],walker.searched_pattern)==0 and walker.x==walker.searched_node:
            fpts.append(walker.t)
            walker.reset()
            break

fig,ax=plt.subplots()
ax.hist(fpts,bins=50)
plt.text(0.7,0.7,'mean={m},std={s}'.format(m=round(np.mean(fpts),2),s=round(np.std(fpts),2)),transform=ax.transAxes)
plt.show()

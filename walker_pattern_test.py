import random_walker as rw
from dag_exps import poisson_ditree,directify
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

#np.random.seed(0)
G,root=poisson_ditree(2.,100)
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

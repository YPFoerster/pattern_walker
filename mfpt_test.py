import random_walker as rw
from dag_exps import poisson_ditree,directify,leaves
import networkx as nx
from networkx.generators import balanced_tree
import numpy as np

r=3 #offspring number
h=5 #height
gamma=0.2 #mutation rate
N=100 #bits in a pattern; must be adapted to ensure uniqueness of patterns
number_of_samples=5000
max_time=5000

name_string='r{r}h{h}gamma{gamma}N{N}'.format(r=r,h=h,gamma=str(round(gamma,2)).replace('.','-'),N=N,number_of_samples=number_of_samples, max_time=max_time)
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

mfpt=walker.get_mfpt()
print(mfpt)

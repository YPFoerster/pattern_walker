import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from networkx.drawing.nx_agraph import graphviz_layout
import networkx as nx
import utils
import random_walker as rw


def make_tree(lam,N,gamma,overlap):
    G,root=utils.poisson_ditree(lam)
    leaves = utils.leaves(G)
    r=len(list(G.successors(root)))
    sections=[i*int(N/r) for i in range(r+1)]
    sections=[(sections[i],min(sections[i+1]+overlap,N)) for i in range(len(sections[:-1]))]
    walker=rw.sectionedPatternWalker(G.copy(),root,N,gamma,sections)
    walker.set_weights()
    return G,root,walker

make_tree=utils.seed_decorator(make_tree,0)


#Create patternWalker based on the above tree.
pattern_len=30 #String/pattern length
flip_rate=5/pattern_len #Bit flipping rate for propagating patterns

offspring_factor=2.
overlap=10
H,root,G=make_tree(offspring_factor,pattern_len,flip_rate,overlap)

temp=list(set(nx.ancestors(H,G.target_node))-set([G.root]))
target_branch=[x for x in temp if x in H.successors(G.root) ]
target_branch=target_branch+list(nx.descendants(H,target_branch[0]))
alpha=list([G.root]+target_branch)
beta=list(set(H.nodes)-set(alpha))

#T=nx.to_numpy_matrix(G,nodelist=[root]+[x for x in H.successors(root)])
T=nx.to_numpy_matrix(G,nodelist=alpha,weight='weight')
print(np.sum(T,axis=-1),T[1,:])

"""
#G=rw.patternWalker(H,root,pattern_len,flip_rate,metric=metric)
#G.set_weights()
pos=graphviz_layout(G,prog='dot') #Positions can be taken from H
(edges,weights) = zip(*nx.get_edge_attributes(G,'weight').items())
nx.draw(G, pos, edgelist=edges, edge_color=weights, edge_cmap=plt.cm.Blues)
nx.draw_networkx_nodes(G,pos,nodelist=target_branch,node_color='y')
nx.draw_networkx_nodes(G,pos,nodelist=[root],node_color='r')
#Mark target node in green.
nx.draw_networkx_nodes(G,pos,nodelist=[G.target_node],node_color='g')
plt.show()
"""

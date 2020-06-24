import random_walker as rw
from dag_exps import poisson_ditree
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

G,root=poisson_ditree(2.,5000)
walker=rw.walker(G,0.2,root)

walker.set_weights()

walker.step()
#A = nx.to_numpy_matrix(walker.G,weight='prob')
#print('Raisin to power!')
#print(np.linalg.matrix_power(A,1000))
"""
print(nx.to_numpy_matrix(walker.G,walker.G.nodes,weight='prob'))
G=walker.G
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos,  node_size = 500)
nx.draw_networkx_edges(G, pos, edge_color='r', arrows=True)
nx.draw_networkx_edge_labels(G,pos,nx.get_edge_attributes(G,'prob'),label_pos=0.3)
plt.show()
"""

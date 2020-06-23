import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from dag_exps import uniform_ditree
from networkx.drawing.nx_agraph import graphviz_layout

G,root=uniform_ditree(10,0)

nx.nx_agraph.write_dot(G,'test.dot')


pos=graphviz_layout(G,prog='dot')


nx.draw(G,pos)
nx.draw_networkx_nodes(G, nodelist=[root],pos=pos,node_color='red')
plt.show()

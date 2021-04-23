import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
from networkx.drawing.nx_agraph import graphviz_layout

__all__ = [
    'plot_tree'
    ]

def plot_tree(G,partition=[]):
    pos=graphviz_layout(G.hierarchy_backup,prog='dot') #positions to display tree "nicely"
    (edges,weights) = zip(*nx.get_edge_attributes(G,'weight').items())
    fig_handle=plt.figure(figsize=(40,20))
    nx.draw(G, pos, edgelist=edges, edge_color=weights, node_size=3000,arrowsize=100,width=5.0, edge_cmap=plt.cm.Blues) #edge colour according to weight
    nx.draw_networkx_nodes(G,pos,nodelist=partition,node_size=5000,node_shape='h',linewidths=10,edgecolors='k')
    nx.draw_networkx_nodes(G,pos,nodelist=[G.root],node_color='r',node_size=3000.) #root in red
    nx.draw_networkx_nodes(G,pos,nodelist=[G.target_node],node_color='g',node_size=3000) #target in green

    return fig_handle,pos

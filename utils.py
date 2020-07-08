"""
Utility functions to generate and modify certain classes of networkx Graphs and
Digraphs.

random_dag-- Generate a random DAG with specified number of nodes and edges.
poisson_ditree-- Generate directed rooted tree with Poissonian offspring.
directify-- Turn tree into directed rooted tree relative to specified root.
leaves-- Return nodes with out-degree zero of given graph.
uniform_ditree-- Draw uniform random tree and directify.

# NOTE: The following four functions were introduced to explore a certain
concepts and are not employed at the moment.
above-- Return number of ancestors.
below-- Return number of descendants.
h-- above-below
w-- above+below
r-- h/w
"""

import networkx as nx
from networkx.utils import generate_unique_node
import numpy as np

def random_dag(nodes, edges):
    """Generate random Directed Acyclic Graph with "nodes" and "edges"."""
    G = nx.DiGraph()
    for i in range(nodes):
        G.add_node(i)
    while edges > 0:
        a = np.random.randint(0,nodes-1)
        b=a
        while b==a:
            b = np.random.randint(0,nodes-1)
        G.add_edge(a,b)
        if nx.is_directed_acyclic_graph(G):
            edges -= 1
        else:
            # we closed a loop!
            G.remove_edge(a,b)
    return G

def poisson_ditree(lam,n_max=100):
    """
    Returns a directed tree with Poissonian branching (distribution with
    parameter 'lam'), terminating at 'n_max' nodes latest.
    """
    G=nx.DiGraph()
    root=generate_unique_node() #all edges will be pointing away from root.
    G.add_node(root)
    current_gen=[root]
    size=1
    new_gen=[]
    #Add children until n_max is exceeded.
    while size<n_max and len(current_gen)>0:
        new_gen=[]
        for node in current_gen:
            offspring = np.random.poisson(lam) #draw number of children
            size+=offspring
            #Create a unique node for every child
            temp=[generate_unique_node() for x in range(offspring)]
            for child in temp:
                new_gen.append(child)
                G.add_edge(node, child)
        current_gen=new_gen.copy()
    return G,root

def directify(G,root):
    """
    Return the DiGraph that is obtained by directing all edges of G "away" from
    "root", i.e. form the node with the smaller to the one with the larger
    smallest distance to "root".

    G-- nx.Graph (or nx.DiGraph); shouldn't have (undirected/weak) cycles
    # TODO: There might still be a way to do this for any DAG.
    # TODO: Check for cycles initally.
    root-- Node in G. In the returned DiGraph, the distance to 'root' will increase
    along every arc.

    return-- tuple (out,root) with "out" the directified nx.Digraph with
    respect to "root", the designated root node.
    """
    out = nx.DiGraph()
    for edge in G.edges(data=False):
        d0=nx.shortest_path_length(G,edge[0],root)
        d1=nx.shortest_path_length(G,edge[1],root)
        if d0<d1:
            out.add_edge(edge[0],edge[1])
        else:
            out.add_edge(edge[1],edge[0])
    return out,root

def leaves(G):
    """
    Return the nodes of G with out-degree zero in a list.

    G-- nx.Digraph

    return-- List of leaves nodes.
    """
    return [node  for (node,od) in G.out_degree() if od==0]

def uniform_ditree(n,seed=None):
    """Draw uniform tree of size "n" and directify relative to random root."""
    G=nx.generators.random_tree(n,seed)
    root = np.random.choice(list(G.nodes))
    G,root=directify(G,root)
    return G,root

def above(G,x):
    return len(nx.ancestors(G,x))

def below(G,x):
    return len(nx.descendants(G,x))

def h(G,x):
    return below(G,x)-above(G,x)

def w(G,x):
    return below(G,x)+above(G,x)

def r(G,x):
    return h(G,x)/w(G,x)

if __name__=="__main__":
"""Tests to come."""

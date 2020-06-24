"""
Generate random directed acyclic graphs and experiment with them.
"""

import networkx as nx
from networkx.utils import generate_unique_node
import numpy as np
import matplotlib.pyplot as plt

def random_dag(nodes, edges):
    """Generate a random Directed Acyclic Graph (DAG) with a given number of nodes and edges."""
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
    root=generate_unique_node()
    G.add_node(root)
    current_gen=[root]
    size=1
    new_gen=[]
    while size<n_max and len(current_gen)>0:
        new_gen=[]
        for node in current_gen:
            offspring = np.random.poisson(lam)
            size+=offspring
            temp=[generate_unique_node() for x in range(offspring)]
            for child in temp:
                new_gen.append(child)
                G.add_edge(node, child)
        current_gen=new_gen.copy()
    return G,root

def directify(G,root):
    """
    Returns the DiGraph that is obtained by directing all edges of G "away" from
    the "root", i.e. form the node with the smaller to the one with the larger
    smallest distance to "root".
    ----------------------------------------------------------------------------
    G: nx.Graph (or nx.DiGraph); shouldn't have (undirected/weak) cycles
    # TODO: There might still be a way to do this for any DAG.
    # TODO: Check for cycles initally.
    root: Node in G. In the returned DiGraph, the distance to 'root' will increase
    along every arc.
    ----------------------------------------------------------------------------
    returns: tuple (out,root) with "out" the nx.Digraph with edges pointing away
    from "root", and the root node.
    """
    out = nx.DiGraph()
    for edge in G.edges(data=False):
        if nx.shortest_path_length(G,edge[0],root)<nx.shortest_path_length(G,edge[1],root):
            out.add_edge(edge[0],edge[1])
        else:
            out.add_edge(edge[1],edge[0])
    return out,root


def uniform_ditree(n,seed=None):
    G_temp=nx.generators.random_tree(n,seed)
    G=nx.DiGraph()
    G.add_nodes_from(G_temp.nodes)
    root = np.random.choice(list(G.nodes))
    for edge in G_temp.edges(data=False):
        if nx.shortest_path_length(G_temp,edge[0],root)<nx.shortest_path_length(G_temp,edge[1],root):
            G.add_edge(edge[0],edge[1])
        else:
            G.add_edge(edge[1],edge[0])

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

    lam=2.
    G,root=poisson_ditree(lam,5000)
    H,_=poisson_ditree(lam,5000)
    GH=nx.DiGraph(G)
    GH.add_edges_from(H.edges)
    print(nx.is_arborescence(G),len(G))
    print(nx.is_arborescence(H),len(H))
    undir_G=nx.Graph(G)
    nodes=len(G.nodes)
    edges=100
    """
    G = random_dag(nodes,edges)
    undir_G = nx.Graph(G)
    """
    check=0
    source=0
    path=[]
    r_array=[]
    a_array=[]
    b_array=[]
    leaves=[x for x in G.nodes() if G.out_degree(x)==0 and G.in_degree(x)==1 or G.in_degree(x)==0]
    sources=np.random.choice(leaves,max(int(len(leaves)/100),1))

    #connection_points_H=np.random.choice(list(H.nodes),max(int(len(H)/10),1))
    #connection_points_G=np.random.choice(list(G.nodes),max(int(len(H)/10),1))
    connection_points_G=np.random.choice([ np.random.choice(list(nx.ancestors(G,source))) for source in sources ],max(int(len(H)/10),1))
    connection_points_H=np.random.choice(list(H.nodes),max(int(len(H)/10),1))


    GH.add_edges_from([ (connection_points_H[i],connection_points_G[i]) for i in range(len(connection_points_G))])


    for source in sources:
    #for source in [leaf]:
        path = nx.ancestors(G,source)
        path=list(path)+[source]
        a_list=np.array([ above(G,node) for node in path])
        idx=np.argsort(a_list)
        path=[path[i] for i in idx]


        #path=nx.ancestors(GH,source)
        a_list=np.array([ above(GH,node) for node in path])
        #a_list=a_list[idx]
        a_array.append(a_list)
        r_array.append( [r(GH,node) for node in path] )
        b_array.append([ below(GH,node) for node in path])
    #a_list=np.mean(a_array,axis=-1)
    #r_list=np.mean(r_array,axis=-1)
    """
    source=np.random.choice(leaves)
    #path = list(nx.node_connected_component(undir_G,source))
    path = nx.ancestors(G,source)
    check=len(path)
    #path = list( nx.descendants(G,source) )+list( nx.ancestors(G,source) )+[source]
    path=list(path)+[source]
    a_list=np.array([ above(G,node) for node in path])
    idx=np.argsort(a_list)
    a_list=a_list[idx]
    path=[path[i] for i in idx]

    b_list=[ below(G,node) for node in path]
    h_list=np.array(b_list)-np.array(a_list)  #[ h(G,node) for node in path ]
    w_list=np.array(b_list)+np.array(a_list)
    r_list=np.array(h_list)/np.array(w_list)
    """

    f,ax = plt.subplots(1,1)
    ax_r=ax.twinx()

    lgd_lines =[]
    lgd_labels=[]
    for row in range(len(a_array)):
        tmp_a=ax.plot(a_array[row],'-b', label='a')
        #tmp_b=ax[0].plot(b_array[column],'-g',label='b')
        tmp_r=ax_r.plot(r_array[row],'xk', label='r')#,label='r')
    lgd_lines.append(*tmp_a)
        #lgd_lines.append(*tmp_b)
    lgd_lines.append(*tmp_r)
    lgd_labels=[l.get_label() for l in lgd_lines]

    ax_r.set_ylabel('$r(x)$')
    ax.set_ylabel('$a(x)$,$b(x)$')
    ax.set_xlabel('Distance of $x$ to root')
    ax.legend(lgd_lines,lgd_labels,loc=0)
    plt.title('Size of tree:{}, Poissonian branching parameter:{}'.format(nodes,lam))
    #ax_r.legend(loc='best')
    #ax[0].legend(loc='best')
    #subG=G.subgraph(path)
    #pos=nx.spring_layout(subG)
    #nx.draw(subG,ax=ax[1],pos=pos)
    #nx.draw(subG,ax=ax[1],nodelist=[root,connection_point_G],pos=pos,node_color=['red','white'])
    plt.show()

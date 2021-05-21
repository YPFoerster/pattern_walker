"""
Utility functions to generate and modify certain classes of networkx Graphs and
DiGraphs.

random_dag-- Generate a random DAG with specified number of nodes and edges.
poisson_ditree-- Generate directed rooted tree with Poissonian offspring.
directify-- Turn tree into directed rooted tree relative to specified root.
sources-- Return nodes with in-degree zero of given graph.
leaves-- Return nodes with out-degree zero of given graph.
uniform_ditree-- Draw uniform random tree and directify.
list_degree_nodes-- Return list of nodes with desired degree.
downward_current-- Calculate downward two-step probabilites through given nodes.
upward_current-- Calculate upward two-step probabilites through given nodes.
path_direction_profile-- Return list of step directions for a path and reference.
largest_eigenvector-- Return the dominant eigenvector of a Markov chain.
normalised_laplacian-- Return normalised_laplacian Laplacian of a Markov chain.
mfpt-- Return mean first passage times between set of Markov chain states.

# NOTE: The following five functions were introduced to explore a certain
concepts and are not employed at the moment.

above-- Return number of ancestors.
below-- Return number of descendants.
h-- above-below
w-- above+below
r-- h/w
"""

import networkx as nx

import numpy as np
from copy import deepcopy
import timeit
from functools import wraps
#Monkey patched uuid.uuid4 to create reproducible unique nodes comparable to networkx.utils.generate_unique_node
#from networkx.utils import generate_unique_node
import uuid
import random
rd = random.Random()
# -------------------------------------------
# Remove this block to generate different
# UUIDs everytime you run this code.
# This block should be right below the uuid
# import.

rd.seed(0)
uuid.uuid4 = lambda: uuid.UUID(int=rd.getrandbits(128))
# --------------------------------------------

__all__ = [
    'random_dag', 'poisson_ditree', 'balanced_ditree', 'directify', 'sources',\
    'leaves', 'uniform_ditree', 'list_degree_nodes', 'downward_current',\
    'upward_current', 'path_direction_profile', 'largest_eigenvector',\
    'normalised_laplacian', 'mfpt', 'block_indices', 'filter_nodes', 'seed_decorator',\
    'spanning_tree_with_root','tree_weight','tree_mfpts', 'cluster_by_branch'
    ]

def random_dag(nodes, edges):
    """
    Generate random Directed Acyclic Graph with "nodes" and "edges".

    Example:
    >>> G=random_dag(20,40)
    >>> len(G)
    20
    >>> len(G.edges())
    40
    """
    G = nx.DiGraph()
    for i in range(nodes):
        G.add_node(i)
    while edges > 0:
        a = np.random.randint(0,nodes)
        b=a
        while a==b or (a,b) in G.edges():
            b = np.random.randint(0,nodes)
        G.add_edge(a,b)
        if nx.is_directed_acyclic_graph(G):
            edges -= 1
        else:
            # we closed a loop!
            G.remove_edge(a,b)
    return G

def poisson_ditree(lam,n_max=100,n_min=3):
    """
    Return a directed tree with Poissonian branching. Terminate branching when
    number of nodes >=n_max. NOTE, number of nodes is not prevented from
    exceeding n_max.

    lam-- Parameter of Poissonian branching distribution.
    n_max-- Size threshold to stop branching.
    n_min-- Lower size threshold

    return-- (G,root), with G the tree and root the first node.

    Example:
    >>> G,root=poisson_ditree(2)
    >>> len(G)>0
    True
    >>> root in G.nodes()
    True
    """
    G=nx.DiGraph()
    root=generate_unique_node() #all edges will be pointing away from root.
    G.add_node(root)
    while len(G)<n_min:
        current_gen=[root]
        new_gen=[]
        #Add children until n_max is exceeded.
        while len(G)<n_max and len(current_gen)>0:
            new_gen=[]
            for node in current_gen:
                offspring = np.random.poisson(lam) #draw number of children
                #Create a unique node for every child
                temp=[generate_unique_node() for x in range(offspring)]
                for child in temp:
                    new_gen.append(child)
                    G.add_edge(node, child)
            current_gen=new_gen.copy()
    return G,root

def balanced_ditree(r,h):
    """
    Generate a perfectly balanced tree and return as nx.Digraph such that egdes
    point away from the root.

    r-- Branching factor. Each node except for leaves has r children.
    h-- Height of the tree. The root sits on level 0, leaves at distance h from
        root.

    return-- (G,root), where root is the unique node in G with degree r (r>1)
        and G is the nx.DiGraph with edges directed towards increasing distance
        from root.

    Example:
    >>> G,root=balanced_directed_tree(3,4)
    >>> type(G)
    <class 'networkx.classes.digraph.DiGraph'>
    >>> G.out_degree(root)==3
    True
    >>> nx.is_arborescence(G)
    True
    """
    G=nx.generators.balanced_tree(r,h)
    root=None
    #root is the only node in G with degree r, all others have degree r+1.
    for node in G.nodes():
        if nx.degree(G,node)==r:
            root=node
            break
    labels={node:generate_unique_node() for node in G.nodes()}
    nx.relabel_nodes(G,labels,copy=False)
    root=labels[root]
    return directify(G,root)

def uniform_ditree(n,seed=None):
    """
    Draw uniform tree of size n and directify relative to random root.

    n-- integer; number of nodes.
    seed-- seed passed to np.random

    return-- G,root

    # NOTE: Bit of a convenience function; might be removed.

    Example:
    >>> G,_ = uniform_ditree(10,0)
    >>> len(G)
    10
    >>> nx.is_arborescence(G)
    True
    """
    G=nx.generators.random_tree(n,seed)
    root = np.random.choice(list(G.nodes))
    G,root=directify(G,root)
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

    Example:
    >>> G=nx.generators.trees.random_tree(100)
    >>> root=np.random.choice(G.nodes())
    >>> H,_=directify(G,root)
    >>> nx.is_arborescence(nx.DiGraph(G))
    False
    >>> nx.is_tree(G)
    True
    >>> nx.is_arborescence(H)
    True
    >>> nx.is_tree(H)
    True
    >>> H.to_undirected().edges()==G.edges()
    True
    >>> # Note that G!=H.to_undirected() even if all edges are identical.
    >>> H.to_undirected()==G
    False
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

def spanning_tree_with_root(G,root,edge_direction='up'):
    out=directify(G,root)
    if edge_direction=='up':
        return nx.reverse(out[0])
    elif edge_direction=='down':
        return out[0]
    else:
        return None
def tree_weight(G,root,edge_direction='up'):
    tree=spanning_tree_with_root(G,root,edge_direction)
    temp=1
    for (u,v) in tree.edges:
        temp*=G.get_edge_data(u,v)['weight']
    return temp


def sources(G):
    """
    Return the nodes of G with in-degree zero in a list.

    G-- nx.Digraph

    return-- List of source nodes.

    Example:
    >>> G=random_dag(10,20)
    >>> sources_list=sources(G)
    >>> len(sources_list)>0
    True
    >>> any([G.in_degree(x)>0 for x in sources_list])
    False
    """
    return [node  for (node,id) in G.in_degree() if id==0]

def leaves(G):
    """
    Return the nodes of G with out-degree zero in a list.

    G-- nx.Digraph

    return-- List of leaves nodes.

    Example:
    >>> G=random_dag(10,20)
    >>> leaves_list=leaves(G)
    >>> len(leaves_list)>0
    True
    >>> any([G.out_degree(x)>0 for x in leaves_list])
    False
    """
    return [node  for (node,od) in G.out_degree() if od==0]

def list_degree_nodes(G,deg,max_num=1):
    """
    Return list of nodes (max_num or fewer) of G with degree equal to deg.

    G-- The graph to be searched.
    deg-- The degree to look for.
    max_num-- When max_num nodes of degree deg are found, the search ends.

    return-- List of found nodes.

    Example:
    >>> G=nx.generators.classic.balanced_tree(3,3)
    >>> len(list_degree_nodes(G,3))
    1
    >>> len(list_degree_nodes(G,5))
    0
    """
    out=[]
    for node in G.nodes():
        if nx.degree(G,node)==deg:
            out.append(node)
            if len(out)>=max_num:
                break
    return out


def downward_current(G,prob_dict,data_key='weight',nodelist=None):
    """
    Iterate through nodelist and return list of two-step probabilities from
    parent node to any of the children for every node in the list. Parents and
    children are determined based on G.

    G-- networkx.Digraph. Does not neccessarily need to be arborescent.
        NOTE! For certain examples (like random_walker.patternWalker),
        G might not be the graph on which a random walker with these two-step
        probabilites treads. The reason is that introducing probabilites for
        stepping upwards changes the notion of "children" and "parents". Make
        sure that the input graph G reflects the desired hierarchy.
    prob_dict-- Edge data for nodes in G in the form dict-of-dict-of-dicts,
        since that is returned by nx.to_dict_of_dicts; data
        interpreted as probabilites.
    data_keyword--The key to access the right datum for each edge in prob_dict.
    nodelist-- The nodes in G to consider (default None-> all nodes).

    return-- list of two-step probabilities.

    Example:
    >>> import random_walker as rw
    >>> G=nx.generators.classic.balanced_tree(3,3)
    >>> root=list_degree_nodes(G,3)[0]
    >>> G,_=directify(G,root)
    >>> G=rw.patternWalker(G,root,flip_rate=0.02,pattern_len=20)
    >>> G.set_weights()
    >>> down=downward_current(G.hierarchy_backup,nx.to_dict_of_dicts(G),'weight')
    >>> len(down)==len(G)
    True
    """
    out=[]
    if nodelist is None:
        nodelist=G.nodes()
    for site in nodelist:
        parents=G.predecessors(site)
        children=G.successors(site)
        #Note that the formula allows for more than one parent.
        out.append(
            np.sum([prob_dict[parent][site][data_key] for parent in parents])*\
            np.sum([prob_dict[site][child][data_key] for child in children])
            )
    return out

def upward_current(G,prob_dict,data_key='weight',nodelist=None):
    """
    As down_current, but in the oppsite direction.
    Iterate through nodelist and return list of two-step probabilities from
    any child node to the parent for every node in the list. Parents and
    children are determined based on G.

    G-- networkx.Digraph. Does not neccessarily need to be arborescent.
        NOTE! For certain examples (like random_walker.patternWalker),
        G might not be the graph on which a random walker with these two-step
        probabilites treads. The reason is that introducing probabilites for
        stepping upwards changes the notion of "children" and "parents". Make
        sure that the input graph G reflects the desired hierarchy.
    prob_dict-- Edge data for nodes in G in the form dict-of-dict-of-dicts,
        since that is returned by nx.to_dict_of_dicts; data
        interpreted as probabilites.
    data_keyword--The key to access the right datum for each edge in prob_dict.
    nodelist-- The nodes in G to consider (default None-> all nodes).

    return-- list of two-step probabilities.

    Example:
    >>> import random_walker as rw
    >>> G=nx.generators.classic.balanced_tree(3,3)
    >>> root=list_degree_nodes(G,3)[0]
    >>> G,_=directify(G,root)
    >>> G=rw.patternWalker(G,root,flip_rate=0.02,pattern_len=20)
    >>> G.set_weights()
    >>> up=upward_current(G.hierarchy_backup,nx.to_dict_of_dicts(G),'weight')
    >>> len(up)==len(G)
    True
    """
    out=[]
    if nodelist is None:
        nodelist=G.nodes()
    for site in nodelist:
        parents=G.predecessors(site)
        children=G.successors(site)
        #Note that the formula allows for more than one parent.
        out.append(
            np.sum([prob_dict[site][parent][data_key] for parent in parents])*\
            np.sum([prob_dict[child][site][data_key] for child in children])
            )
    return out

def path_direction_profile(G,reference,path):
    """
    For each node in path, decide whether the distance to the reference has
    increased or decreased.

    G-- nx.Graph or anothing compatible with nx.shortest_path_length.
    reference-- The based on which 'direction' is defined. A step that
        decreases the distance to reference is assigned -1, else +1.
    path-- Iterable of nodes, representing a path. Does not neccessarily need
        to be an list of successively adjecent nodes.

    return-- list of +1 and -1, indicating the direction of each step. Note
        that len(return)=len(path)-1.

    Example:
    >>> G=nx.generators.classic.balanced_tree(1,10)
    >>> root=list_degree_nodes(G,1)[0]
    >>> path=np.random.choice(G.nodes(),20,replace=True)
    >>> profile=path_direction_profile(G,root,path)
    >>> len(profile)
    19
    """
    return [
            -1 if
            nx.shortest_path_length(G,path[i+1],reference)>
            nx.shortest_path_length(G,path[i],reference)
            else 1
            for i in range(len(path[:-1]))
            ]

def largest_eigenvector(G,weight='weight'):
    """
    Return the maximum eigenvector of a Markov chain with state space G.

    G-- nx.Graph
    weight-- Name of the edge attibute to use in the transition matrix of the
        Markov chain. Treat matrix entry [i,j] as the transition probability
        from i to j, and assume it to be normalised. (default: 'weight')

    return-- Left eigenvector of transition matrix corresponding to eigenvalue
        with greatest real part. Normalise entries to sum up to unity.

    Example:
    >>> G=nx.generators.classic.balanced_tree(1,10)
    >>> root=list_degree_nodes(G,1)[0]
    >>> # For this G, the largest eigenvector is the stationary
    >>> # distribution of a random walker, all entries positive.
    >>> pi=largest_eigenvector(G,weight=None)
    >>> all( [x>0 for x in pi] )
    True
    """
    #trans_{i,j}=Prob(j|i)= Prob(i->j)
    trans = nx.to_numpy_matrix(G,weight=weight)
    evls, evcs = np.linalg.eig(trans.T)
    max_evl_ndx=np.argmax(np.real(evls))
    #normalise entries to sum up to 1
    evcs[:,max_evl_ndx]/=np.sum(evcs[:,max_evl_ndx])
    #If looking for stationary distribution, entries must be non-negative
    if any( [x<0 for x in evcs[:,max_evl_ndx]] ):
        evcs[:,max_evl_ndx]*=-1
    return np.squeeze(np.array(evcs[:,max_evl_ndx]))

def normalised_laplacian(G,weight='weight',stat_dist=None):
    """
    Return the normalised Laplacian of a Markov chain with state space G.

    G-- nx.Graph
    weight-- Name of the edge attibute to use in the transition matrix of the
        Markov chain. Treat matrix entry [i,j] as the transition probability
        from i to j, and assume it to be normalised. (default: 'weight')
    stat_dist-- Stationary distribution of the Markov chain (left eigenvector
        for the unit eigenvalue of the transition matrix). If None, calculate
        using largest_eigenvector.

    return-- Normalised Laplacian, np.ndarray with shape (len(G),len(G)).

    Example:
    >>> G=nx.generators.classic.balanced_tree(1,10)
    >>> root=list_degree_nodes(G,1)[0]
    >>> L=normalised_laplacian(G,None)
    """
    pi=stat_dist
    if pi is None:
        pi=largest_eigenvector(G,weight=weight)
    trans=nx.to_numpy_matrix(G, weight=weight)
    return np.matmul(np.matmul(np.diagflat(np.sqrt(pi)),np.eye(len(pi))-trans),np.diagflat(1/np.sqrt(pi)))

def mfpt(
    G,node_pairs,weight_str='weight',stat_dist=None,norm_laplacian=None,
    method='grounded_Laplacian'
    ):
    """
    Return mean first passage times (MFPT) of a Markov chain on G
    for list of pairs of nodes.

    G-- nx.Graph.
    node_pairs-- Iterateable with node tuples (start_node,target_node).
    weight-- Name of the edge attibute to use in the transition matrix of the
        Markov chain. Treat matrix entry [i,j] as the transition probability
        from i to j, and assume it to be normalised. (default: 'weight')
    stat_dist-- Stationary distribution of the Markov chain (left eigenvector
        for the unit eigenvalue of the transition matrix). If None, calculate
        using largest_eigenvector.
    norm_laplacian-- Normalised Laplacian of the Markov chain. If None,
        calculate using norm_laplacian.
        (Only needed for method='fundamental_matrix')
    method-- Either 'fundamental_matrix' (default) or 'eig'. The former uses
        the (pseudo)inverse of the normalised Laplacian
        (see Yanhua & Zhang, 2010) the latter eigenvector decomposition of the
        transition matrix. Note that the letter requires DB to be correct, and
        NOTE THAT THE LATTER METHOD IS FAULTY AT THE MOMENT.

    return-- If node_pairs contains only one tuple (start,target):
        MFPT between start and target.
        Otherwise: Dict of dicts of MFPTs keyed by the nodes in node_pairs.

    Example:
    >>> import random_walker as rw
    >>> G=nx.generators.classic.balanced_tree(3,3)
    >>> root=list_degree_nodes(G,3)[0]
    >>> G,_=directify(G,root)
    >>> G=rw.patternWalker(G,root,flip_rate=0.02,pattern_len=20)
    >>> G.set_weights()
    >>> target=np.random.choice(G.nodes())
    >>> while target==root:
    ...    target=np.random.choice(G.nodes())
    >>> t=mfpt(G,[(root,target)],weight_str='weight')
    >>> t>0
    True
    """
    out={}
    if method=='fundamental_matrix':
        trans=nx.to_numpy_matrix(G, weight=weight_str)
        pi=stat_dist
        nlap=norm_laplacian
        if pi is None:
            pi=largest_eigenvector(G,weight=weight_str)
        if norm_laplacian is None:
            norm_laplacian=normalised_laplacian(G,weight_str,pi)
        inv_nlap=np.linalg.pinv(norm_laplacian) #pseudoinverse of nlap
        for pair in node_pairs:
            start_ndx=list(G.nodes()).index(pair[0])
            target_ndx=list(G.nodes()).index(pair[1])
            out[pair]=inv_nlap[target_ndx,target_ndx]/pi[target_ndx]-\
                inv_nlap[start_ndx,target_ndx]/np.sqrt(pi[start_ndx]*pi[target_ndx])

    if method=='Kirkland':
        a,b=block_indices(G,G.target_node)
        nodes=list(G.nodes())
        alpha=[nodes.index(x) for x in a]
        beta=[nodes.index(x) for x in b]
        node_pair_inds={(node_1,node_2): (a.index(node_1), a.index(node_2)) for (node_1,node_2) in node_pairs }
        W=nx.to_numpy_array(G,nodelist=a+b,weight=weight_str)
        W_alpha=W[:len(a),:len(a)]
        W_beta=W[-len(b):,-len(b):]
        W_ab=W[:len(alpha),-len(b):]
        W_ba=W[-len(b):,:len(a)]
        u=np.linalg.inv(np.eye(len(G)-1)-W[1:,1:])
        pi=np.concatenate(([1],-np.matmul(-W[0,1:],u).T))
        pi=pi/np.sum(pi)
        X=u[:len(a)-1,:len(a)-1]#np.linalg.inv(np.eye(len(a)-1)-W_alpha[1:,1:])
        Y=u[-len(b):,-len(b):]
        h=-W_alpha[0,1]*X[0,:]
        H=np.matmul(np.ones((len(a)-1,1)),np.expand_dims(h,axis=1).T)
        delta=np.sum(W_alpha[0,:])*np.sum(np.linalg.matrix_power(X,2)[0,:])
        beta=1+np.sum(W_alpha[0,:])*np.sum(X[0,:])
        F=X-delta/beta*np.eye(len(a)-1)
        y=np.sum(np.matmul(W_ab,Y))
        Q=np.zeros((len(a),len(a))) #Q is the gerneralised group inverse of 1-P_alpha
        Q[0,0]=delta/beta**2
        Q[0,1:] = -W_alpha[0,1]*1/beta*np.array([ np.linalg.matrix_power(X,2)[0,j]-delta/beta*X[0,j] for j in range(len(X))])
        Q[1:,0] = -1/beta*np.squeeze(np.sum(F, axis=1))
        Q[1:,1:]=X+1/delta*(np.matmul(np.matmul(X,H),X) - np.matmul(np.matmul(F,H),F))
        Qdg = np.diag(np.diag(Q))
        gamma_alpha=1/np.sum(pi[:len(a)])
        pi_alpha=pi[:len(a)]*gamma_alpha
        pi_alpha=np.squeeze(np.array([x for x in pi_alpha]))
        MQ = np.matmul(np.eye(len(a))-Q+np.matmul(np.ones((len(a),len(a))),Qdg),np.diag(1/pi_alpha))
        V_alpha=np.zeros((len(a),len(a)))
        FJ=np.matmul(F,np.ones((len(a)-1,len(a)-1)))
        V_alpha[0,1:]=delta/beta**2+1/beta*FJ[:,0].T
        V_alpha[1:,0]=-delta/beta**2-1/beta*np.squeeze(FJ[:,0])
        V_alpha[1:,1:]=-1/beta*(FJ-FJ.T)
        V_alpha*=y
        M=gamma_alpha*MQ+V_alpha
        for pair in node_pairs:
            start=node_pair_inds[pair][0]
            target=node_pair_inds[pair][1]
            out[pair]=M[ start,target ]

    if method=='grounded_Laplacian':
        for pair in node_pairs:
            node_order=[pair[0]]+list(set( list(G.nodes) )-set(pair)  )+[pair[1]]
            W=nx.to_numpy_array(G, weight=weight_str,nodelist=node_order)
            if (np.sum(W,axis=-1)!=1).any:
                W=np.diag(1/np.sum(W,axis=-1)).dot(W)
            out[pair]=np.sum( np.linalg.inv( np.eye(len(G)-1)-W[:-1,:-1] ),axis=-1 )[0]

    if method=='eig':
        # NOTE: Not fixed yet.
        trans = nx.to_numpy_matrix(G,weight=weight_str).T #trans_{i,j}=Prob(i|j)= Prob(j->i)
        evls, r_evcs = np.linalg.eig(trans)
        max_evl_ndx=np.argmax(np.real(evls))
        r_evcs[:,max_evl_ndx]/=np.sum(r_evcs[:,max_evl_ndx])
        if any( [x<0 for x in r_evcs[:,max_evl_ndx]] ):
            r_evcs[:,max_evl_ndx]*=-1
        p_eq=r_evcs[:,max_evl_ndx]
        l_evcs=np.linalg.inv(r_evcs)
        root_ndx=list(G.nodes()).index(self.root)
        target_ndx=list(G.nodes()).index(self.target_node)
        for pair in node_pairs:
            start_ndx=list(G.nodes()).index(pair[0])
            target_ndx=list(G.nodes()).index(pair[1])
            out[pair[0]]={
                pair[1]:1/p_eq[start_ndx]*(
                    1+ np.sum([
                        evls[l]/(1-evls[l])*r_evcs[start_ndx,l]*\
                        (l_evcs[start_ndx,l]-l_evcs[target_ndx,l])
                        for l in range(len(evls)) if l != max_evl_ndx
                        ])
                    )
                }
    if len(node_pairs)==1:
        return out[node_pairs[0]]
    else:
        return out

def tree_mfpts(tree,start,target):
    #Find MFPT based on tree weights. Tree has to be an arborescence.
    sigma_target=tree_weight(tree,target)
    shortest_path = nx.shortest_path(tree,start,target)
    sigma_start_target=0

    for cut_node_ndx in range(len(shortest_path)-1):
        temp_sigma=1
        G_cut=deepcopy(tree)
        G_cut.remove_edge(shortest_path[cut_node_ndx],shortest_path[cut_node_ndx+1])
        G_cut.remove_edge(shortest_path[cut_node_ndx+1],shortest_path[cut_node_ndx])
        comps=nx.weakly_connected_components(G_cut)
        start_comp=deepcopy(G_cut)
        target_comp=deepcopy(G_cut)
        for comp in comps:
            if not (start in comp):
                start_comp.remove_nodes_from(comp)
            elif not (target in comp):
                target_comp.remove_nodes_from(comp)
        temp_sigma*=tree_weight(target_comp,target)
        temp_sigma*=np.sum([tree_weight(start_comp,node) for node in start_comp])
        sigma_start_target+=temp_sigma
    return sigma_start_target/sigma_target

def block_indices(G,node):
    """
    Return NODE lists for the block containing node and the relevant other block,
    determined based on the hierarchy_backup of G. The former block contains the
    root.
    ## TODO: Change name, because we don't return indices. Go through importing scripts and correct usage.

    G: rw.patternWalker
    node: node in G other than G.root
    """
    temp=list(set(nx.ancestors(G.hierarchy_backup,node))-set([G.root]))#everything upstream of node (we have to remove the root)
    branch=[x for x in temp if x in G.hierarchy_backup.successors(G.root) ]#those nodes upstream of node that are direct children of G.root
    branch=branch+list(nx.descendants(G.hierarchy_backup,branch[0]))#everything downstream of those nodes we have just found (if we hadn't removed the root two lines above, this would be everthing)
    alpha=list([G.root]+branch)#add the root again
    beta=list(set(G.nodes)-set(alpha))#the complement of alpha
    return alpha, beta

def cluster_by_branch(G):
    """
    Returns a list of nodes for every node on the shortest path between G.root
    and G.target_node such that the list for each node contains all descendants
    of the node NOT following the shortest path to the target. """
    path = nx.shortest_path(G,G.root, G.target_node)
    path_edges={path[i]:[(path[i],path[i+1]),(path[i+1],path[i])] for i in range(len(path)-1)}
    H=deepcopy(G.hierarchy_backup)
    clusters={}
    for node in path:
        try:
            H.remove_edges_from(path_edges[node])
            clusters[node]={'cluster':list(nx.descendants(H,node))+[node]}
        except KeyError:
            if node==G.target_node:
                clusters[node]={'cluster':[node]}
    return clusters

def filter_nodes(G,attrstr,value):
    return [node for node,attrdict in G.nodes.items() if attrdict[attrstr]==value]


def timer_decorator(func,args,kwargs):
    @wraps(func)
    def wrapper():
        return func(*args,*kwargs)
    return wrapper


def seed_decorator(func,seed=0):
    """Set fixed seed in front of function and reset afterwards.
    # TODO: Can this done more elegantly? probably..."""
    def actual_seed_decorator(func):

        @wraps(func)
        def wrapper(*args,**kwargs):
            np.random.seed(seed)
            out=func(*args,**kwargs)
            np.random.seed()
            return out
        return wrapper
    return actual_seed_decorator(func)

def generate_unique_node():
    """
    The networkx.utils function uses uuid.uuid1(). With uuid4() it is easier to
    make the results reproducible.
    """
    return str(uuid.uuid4())


if __name__=="__main__":
    import doctest
    doctest.testmod()

    if True:
        from networkx.generators.classic import balanced_tree
        import random_walker as rw
        G=balanced_tree(3,3)
        root=list_degree_nodes(G,3)[0]
        G,_=directify(G,root)
        G=rw.walker(G,root,1)
        G.set_weights()
        args=[G,[(root,x) for x in leaves(G)]]
        kwargs={'weight_str': 'weight'}
        time_func=timer_decorator(mfpt,args,kwargs)
        t=timeit.timeit(stmt=time_func,number=1000)
        print(t)

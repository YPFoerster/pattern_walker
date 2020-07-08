"""
Asymmetric random walkers on a directed graph and some useful functions.

walker-- Basic biased random walker derived from nx.Digraph.
patternWalker-- Random walker derived from walker, searching a binary pattern.
hamming_dist-- Return Hamming distance of two binary strings.
mutate_pattern-- Flip every entry in a binary string with given probability.
count_pattern_duplicates-- Return number of duplicate patterns of a
    patternWalker.
"""

import numpy as np
import networkx as nx
from utils import poisson_ditree

class walker(nx.DiGraph):
    """
    A biased random walker on a Digraph.

    Methods:
    get_probs-- Return transition probabilities from given node.

    set_weights-- Distribute probabilites over edges of the graph.

    step-- From current position, take a step to a random neighbour.

    reset-- reset current position to root, delete history and time.

    Variables:
    bias-- Prefactor for probability of stepping upwards instead of downwards.

    trace-- List of nodes visited in the past.

    x-- current position of the walker.

    t-- Number of steps taken.

    Example:
    >>> G,root=poisson_ditree(2)
    >>> w=walker(G,root,0.2)
    >>> w.set_weights()
    >>> for i in range(5):
    ...     w.step()
    ...
    >>> w.reset()
    >>> assert len(w.trace)==1
    """

    def __init__(self,G,init_pos,bias):
        """Initialise Digraph as G and instantiate class variables."""
        super(walker, self).__init__()
        self.add_edges_from(G.edges())
        self.bias=bias
        self.trace=[init_pos]
        self.x=init_pos
        self.t=0

    def get_probs(self,site):
        """
        Calculate transition probabilites for the walker. All children of site
        receive the same probability, while all parents (one, for a tree)
        get assigned that value multiplied by self.bias.

        site-- The node for that to calculate the probabilities.
        return-- tuple (children,parents,probs), where probs is a dictionary
        keyed by the nodes in children and in parents.
        """
        children=list(self.successors(site))
        parents=list(self.predecessors(site))
        probs={}
        for child in children:
            probs[child]=1./(self.bias*len(parents)+len(children))
        for parent in parents:
            probs[parent]=self.bias/(self.bias*len(parents)+len(children))
        return children,parents,probs


    def set_weights(self):
        """
        Iterate over all nodes and add probabilites (self.get_probs) as
        attributes to edges. Edges are added if they don't already exists, like
        for the upward edge between child and parent.
        """
        for site in self.nodes:
            probs={}
            parents=[]
            children=[]
            children,parents,probs=self.get_probs(site=site)
            nx.set_edge_attributes(self,{(site,child):{'prob':probs[child]} for child in children})
            self.add_edges_from( [ (site,parent,{'prob': probs[parent]}) for parent in parents ] )


    def step(self):
        """Retrieve transition probabilites at self.x to step to a neighbour.
        Append new node to self.trace and increase self.t by one."""
        out_edges_list=self.out_edges(self.x,'prob')
        self.x = np.random.choice( [x[1] for x in out_edges_list],p=[x[2] for x  in out_edges_list] )
        self.trace.append(self.x)
        self.t+=1


    def reset(self):
        """Reset x,trace and t to initial values."""
        self.x=self.trace[0]
        self.trace=[self.x]
        self.t=0

class patternWalker(walker):
    """
    A random walker on a Digraph derived from the walker-class. Transition
    probabilites are based on binary strings assigned to every node.

    Overriding methods:
    get_probs-- Return transition probabilities from given node. Overrides
        walker.get_probs.

    Extending methods:
    set_patterns-- Propagate binary strings from root to all nodes.

    Extending variables:

    pattern_len-- Lenght of binary strings assined to all nodes.

    flip_rate-- Probability to change any string bit from one node to its child.

    root-- The root node, initial position of the walker and string propagation.

    metric-- Function to calculate distances between strings.

    target_node-- Target of the random walker.

    target_pattern-- string of target_node.

    hierarchy_backup-- backup the input graph, because set_weights adds edges.

    Example:
    >>> G,root=poisson_ditree(2)
    >>> w=patternWalker(G,root,20,0.005)
    >>> w.set_weights()
    >>> for i in range(5):
    ...     w.step()
    ...
    >>> w.reset()
    >>> assert len(w.trace)==1
    """

    def __init__(self,G,init_pos,pattern_len,flip_rate,metric=None,search_for=None):
        """
        Initialise variables as described, passing G and it's root to the
        the superclass. Calls set_patterns to assign binary strings to all
        nodes.

        G-- Graph data, must be compatible with class walker.
        init_pos-- inital postion of the walker. Will be handled as root of G
        pattern_len-- Length of binary strings assigned to nodes.
        flip_rate-- Probability of changing any bit in the string propagated
            from parent to child node.
        metric-- Metric for binary strings. (if None: Hamming distance)
        search_for-- The target node of the walker. If None, one is chosen
            randonly from the leaf nodes.
        """
        #Remember to pass walker.bias=1 in super
        super(patternWalker,self).__init__(G,init_pos,1.)
        self.pattern_len=pattern_len
        self.flip_rate=flip_rate
        self.root=init_pos
        if metric is None:
            self.metric=hamming_dist
        self.hierarchy_backup=dict(self.edges())

        self.set_patterns()
        if search_for is None:
            self.target_node=np.random.choice(self.nodes)
            self.target_pattern=self.nodes[self.target_node]['pattern']
        else:
            self.target_node=search_for
            self.target_pattern=self.nodes[self.target_node]['pattern']

    def set_patterns(self):
        """
        Assigns a binary string to every node in the graph, successively by
        generation/level. A pattern/string is generated by based on its parent
        string by the function mutate_pattern.
        """
        self.nodes[self.root]['pattern']=list(np.random.randint(0,2,self.pattern_len))
        last_patterned_gen=[self.root]
        queue=list(self.successors(self.root))
        while len(queue)>0:
            for node in queue:
                pattern=self.nodes[list(self.predecessors(node))[0]]['pattern']
                self.nodes[node]['pattern']=mutate_pattern( pattern,self.flip_rate )
            queue=[suc for node in queue for suc in self.successors(node)]


    def get_probs(self,site):
        """
        THIS METHOD OVERRIDES THE METHOD OF THE SAME NAME IN WALKER.
        Steping probabilities from site to its neighbours. The probabilites are
        inversely proportional to the metric between the target pattern and the
        relevant neighbour's pattern.

        site-- node in the graph.
        return-- tuple (children,parents,probs), where the dictionary probs
        contains the stepping probabilites and is keyed by the entries of
        children and parents.
        """
        # TODO: One could inherit this method of "walker" and isolate the actual
        # probability assignments
        children=list(self.successors(site))
        parents=list(self.predecessors(site))
        probs={}
        for child in children:
            probs[child]=1/(1+self.metric( self.nodes[child]['pattern'],\
                self.target_pattern ))
        for parent in parents:
            probs[parent]=1/(1+self.metric( self.nodes[parent]['pattern'],\
                self.target_pattern ))
        denominator = sum( probs.values() )
        for key in probs.keys():
            probs[key]/=denominator
        return children,parents,probs

    def direction_profile(self):
        """Go through trace of the walker and mark upward and downward steps."""
        return [ -1 if nx.shortest_path_length(self,self.trace[i+1],self.root)>nx.shortest_path_length(self,self.trace[i],self.root) else 1 for i in range(len(self.trace[:-1])) ]

    def get_stat_distr(self):
        trans = nx.to_numpy_matrix(self,weight='prob').T #trans_{i,j}=Prob(i|j)= Prob(j->i)
        evls, r_evcs = np.linalg.eig(trans)
        max_evl_ndx=np.argmax(np.real(evls))
        r_evcs[:,max_evl_ndx]/=np.sum(r_evcs[:,max_evl_ndx])
        if any( [x<0 for x in r_evcs[:,max_evl_ndx]] ):
            r_evcs[:,max_evl_ndx]*=-1
        return np.squeeze(np.array(r_evcs[:,max_evl_ndx]))

    def get_norm_laplacian(self):
        pi=self.get_stat_distr()
        trans=nx.to_numpy_matrix(self.G, weight='prob')
        return np.matmul(np.matmul(np.diagflat(np.sqrt(pi)), np.eye(len(pi))-trans), np.diagflat(1/np.sqrt(pi)))

    def get_YZ_hitting_time(self):
        """
        Following Yanhua and Zhang (2010)
        """
        pi=self.get_stat_distr()
        trans=nx.to_numpy_matrix(self.G, weight='prob')
        l_plus=np.linalg.pinv(
            np.matmul(np.matmul(np.diagflat(np.sqrt(pi)), np.eye(len(pi))-trans), np.diagflat(1/np.sqrt(pi)))
        )
        root_ndx=list(self.G.nodes()).index(self.root)
        target_ndx=list(self.G.nodes()).index(self.target_node)
        #return l_plus[target_ndx,target_ndx]/pi[target_ndx]-l_plus[root_ndx,target_ndx]/np.sqrt(pi[root_ndx]*pi[target_ndx])
        return np.array([l_plus[root_ndx,root_ndx]/pi[root_ndx]-l_plus[target_ndx,root_ndx]/np.sqrt(pi[root_ndx]*pi[target_ndx]) for target_ndx in range(len(self.G))])

    def get_mfpt(self):
        trans = nx.to_numpy_matrix(self.G,weight='prob').T #trans_{i,j}=Prob(i|j)= Prob(j->i)
        evls, r_evcs = np.linalg.eig(trans)
        max_evl_ndx=np.argmax(np.real(evls))
        r_evcs[:,max_evl_ndx]/=np.sum(r_evcs[:,max_evl_ndx])
        if any( [x<0 for x in r_evcs[:,max_evl_ndx]] ):
            r_evcs[:,max_evl_ndx]*=-1
        p_eq=r_evcs[:,max_evl_ndx]
        l_evcs=np.linalg.inv(r_evcs)
        root_ndx=list(self.G.nodes()).index(self.root)
        target_ndx=list(self.G.nodes()).index(self.target_node)
        return np.squeeze(np.array([1/p_eq[root_ndx]*(
        1+ np.sum(
            [
            evls[l]/(1-evls[l])*\
            r_evcs[root_ndx,l]*(l_evcs[root_ndx,l]-l_evcs[target_ndx,l])
             for l in range(len(evls)) if l != max_evl_ndx
            ]
            )
        ) for target_ndx in range(len(self.G))]))

def hamming_dist(a,b):
    temp=[a[i]-b[i] for i in range(len(a))]
    return np.count_nonzero(temp)

def mutate_pattern(pattern,gamma):
    return [ 1-x if np.random.random()<gamma else x for x in pattern ]

def count_pattern_duplicates(pw):
    patterns = list(nx.get_node_attributes(pw.G,'pattern').values())
    duplicates = []
    uniques = []
    for x in patterns:
        if x not in uniques:
            uniques.append(x)
        else:
            duplicates.append(x)
    return len(duplicates)



if __name__=="__main__":
    import doctest
    doctest.testmod()

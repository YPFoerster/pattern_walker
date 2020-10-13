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
import utils

class walker(nx.DiGraph):
    """
    A biased random walker on a Digraph.

    Methods:
    get_probs-- Return transition probabilities from given node.

    set_weights-- Distribute probabilites over edges of the graph.

    step-- From current position, take a step to a random neighbour.

    reset_walker-- reset current position to root, delete history and time.

    Variables:
    bias-- Prefactor for probability of stepping upwards instead of downwards.

    trace-- List of nodes visited in the past.

    x-- current position of the walker.

    t-- Number of steps taken.

    Example:
    >>> from networkx.generators.classic import balanced_tree
    >>> G=balanced_tree(1,10,create_using=nx.DiGraph)
    >>> root=0
    >>> w=walker(G,root,0.2)
    >>> w.set_weights()
    >>> for i in range(20):
    ...     w.step()
    >>> len(w.trace)
    21
    >>> w.reset_walker()
    >>> w.trace
    [0]
    >>> w.trace==[w.x]
    True
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
        for site in self.nodes():
            probs={}
            parents=[]
            children=[]
            children,parents,probs=self.get_probs(site=site)
            nx.set_edge_attributes(
                self,{(site,child):{'weight':probs[child]} for child in children}
                )
            self.add_edges_from(
                [(site,parent,{'weight': probs[parent]}) for parent in parents]
                )


    def step(self):
        """Retrieve transition probabilites at self.x to step to a neighbour.
        Append new node to self.trace and increase self.t by one."""
        out_edges_list=self.out_edges(self.x,'weight')
        self.x = np.random.choice(
            [x[1] for x in out_edges_list],p=[x[2] for x  in out_edges_list]
            )
        self.trace.append(self.x)
        self.t+=1


    def reset_walker(self):
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

    count_pattern_duplicates-- Count strings that appear more than once.

    Extending variables:

    pattern_len-- Lenght of binary strings assined to all nodes.

    flip_rate-- Probability to change any string bit from one node to its child.

    root-- The root node, initial position of the walker and string propagation.

    metric-- Function to calculate distances between strings.

    target_node-- Target of the random walker.

    target_pattern-- string of target_node.

    hierarchy_backup-- backup the input graph, because set_weights adds edges.


    Example:
    >>> from networkx.generators.classic import balanced_tree
    >>> G=balanced_tree(1,10,create_using=nx.DiGraph)
    >>> root=0
    >>> w=patternWalker(G,root,20,0.005)
    >>> 0<= w.num_pattern_duplicates()
    True
    >>> w.set_weights()
    >>> for i in range(20):
    ...     w.step()
    >>> len(w.trace)
    21
    >>> w.reset_walker()
    >>> w.trace
    [0]
    >>> w.trace==[w.x]
    True
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
        self.hierarchy_backup=G.copy()
        super(patternWalker,self).__init__(G,init_pos,1.)
        self.pattern_len=pattern_len
        self.flip_rate=flip_rate
        self.root=init_pos
        if metric is None:
            self.metric=hamming_dist
        else:
            self.metric=metric
        self.set_patterns()
        if search_for is None:
            self.target_node=np.random.choice(utils.leaves(self.hierarchy_backup))
            self.target_pattern=self.nodes[self.target_node]['pattern']
        else:
            self.target_node=search_for
            self.target_pattern=self.nodes[self.target_node]['pattern']

    def set_patterns(self):
        """
        Assigns a binary string to every node in the graph, successively by
        generation/level. A pattern/string is generated based on its parent
        string by the function mutate_pattern.
        """
        self.nodes[self.root]['pattern']=list(np.random.randint(
                                                        0,2,self.pattern_len
                                                        ))
        last_patterned_gen=[self.root]
        queue=list(self.successors(self.root))
        while len(queue)>0:
            for node in queue:
                pattern=self.nodes[list(self.predecessors(node))[0]]['pattern']
                self.nodes[node]['pattern']=mutate_pattern(
                                                pattern,self.flip_rate
                                                )
            queue=[suc for node in queue for suc in self.successors(node)]

    def reset_patterns(self):
        """
        Like set_patterns, but first resets edge data, because set_weights
        introduces edges blurring the hierarchy on that set_patterns relies.
        """
        self.clear()
        self.add_edges_from(self.hierarchy_backup.edges())
        self.set_patterns()
        self.set_weights()

    def num_pattern_duplicates(self):
        """Count all strings that appear more than once."""
        patterns = list(nx.get_node_attributes(self,'pattern').values())
        duplicates = []
        uniques = []
        for x in patterns:
            if x not in uniques:
                uniques.append(x)
            else:
                duplicates.append(x)
        return len(duplicates)

    def get_probs(self,site):
        """
        THIS METHOD OVERLOADS THE METHOD OF THE SAME NAME IN WALKER.
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


class sectionedPatternWalker(patternWalker):
    """
    Example:
    >>> from networkx.generators.classic import balanced_tree
    >>> G=balanced_tree(1,10,create_using=nx.DiGraph)
    >>> root=0
    >>> w=sectionedPatternWalker(G,root,20,0.005)
    >>> 0<= w.num_pattern_duplicates()
    True
    >>> w.set_weights()
    >>> for i in range(20):
    ...     w.step()
    >>> len(w.trace)
    21
    >>> w.reset_walker()
    >>> w.trace
    [0]
    >>> w.trace==[w.x]
    True
    """

    def __init__(self,G,init_pos,pattern_len,flip_rate,sections,metric=None,search_for=None):

        self.sections=self.sections_prep(G,init_pos,pattern_len,sections)
        self.num_sections=len(self.sections)
        super(sectionedPatternWalker,self).__init__(G,init_pos,pattern_len,flip_rate,metric,search_for)


    def set_patterns(self):
        """
        THIS METHOD OVERLOADS THE METHOD OF THE SAME NAME IN PatternWalker.
        Assigns a binary string to every node in the graph, successively by
        generation/level. A pattern/string is generated based on its parent
        string by the function mutate_pattern.
        """

        queue=list(self.successors(self.root))
        self.nodes[self.root]['pattern']=list(np.random.randint(
                                                        0,2,self.pattern_len
                                                        ))
        last_patterned_gen=[self.root]

        while len(queue)>0:
            for node in queue:
                pattern=self.nodes[list(self.predecessors(node))[0]]['pattern']
                self.nodes[node]['pattern']=mutate_pattern(
                                                pattern,self.flip_rate
                                                )
            queue=[suc for node in queue for suc in self.successors(node)]

        principle_branches=list(self.successors(self.root))
        for i in range(self.num_sections):
            self.nodes[principle_branches[i]]['pattern']=[ 0 if ind < self.sections[i][0] or ind>=self.sections[i][1] else self.nodes[principle_branches[i]]['pattern'][ind] for ind in range(self.pattern_len) ]
            #self.nodes[principle_branches[i]]['pattern'][:section_boundaries[i]]=[0]*section_boundaries[i]
            #self.nodes[principle_branches[i]]['pattern'][section_boundaries[i+1]:section_boundaries[-1]]=[0]*(section_boundaries[i+1]-section_boundaries[-1])

            for node in nx.descendants(self,principle_branches[i]):
                self.nodes[node]['pattern']=[ 0 if ind < self.sections[i][0] or ind>=self.sections[i][1] else self.nodes[node]['pattern'][ind] for ind in range(self.pattern_len) ]
                #self.nodes[node]['pattern'][:section_boundaries[i]]=[0]*section_boundaries[i]
                #self.nodes[node]['pattern'][section_boundaries[i+1]:section_boundaries[-1]]=[0]*(section_boundaries[i+1]-section_boundaries[-1])

    def sections_prep(self,G,init_pos,pattern_len,sections):
        out=[]
        if isinstance(sections, list):
            # TODO: More checks recommended to ensure that number of sections is compatible with the hierarchy,that endpoints are included and that we don't overshoot.
            if isinstance(sections[0],int):
                for ind in range(len(sections)-1):
                    out.append( (sections[ind],sections[ind+1]) )
            elif isinstance(sections[0], tuple):
                out=sections

            out[-1]=(out[-1][0],pattern_len)
            sections_compatible=len(list(G.successors(init_pos)))-len(out)
            if sections_compatible>0:
                out=out+sections_compatible*[out[-1]]
            elif sections_compatible<0:
                out=out[:sections_compatible]
                out[-1]=(out[-1][0],pattern_len)
        return out

def hamming_dist(a,b):
    """Return number of non-equal entries of a and b (truncates at len(a))."""
    temp=[a[i]-b[i] for i in range(len(a))]
    return np.count_nonzero(temp)

def mutate_pattern(pattern,gamma):
    """Expect a binary string and flip every entry with probability gamma."""
    return [ 1-x if np.random.random()<gamma else x for x in pattern ]

def sections_by_overlap(pattern_len,num_sections,frac_overlap):
    section_len=int(pattern_len/num_sections)
    shift=int(frac_overlap*section_len/2)
    print(shift,section_len)
    sections=[ ( i*section_len-shift,(i+1)*section_len+shift ) for i in range(num_sections)]
    sections[0]=(0,section_len+shift)
    sections[-1]=((num_sections-1)*section_len-shift,pattern_len)
    return sections



if __name__=="__main__":
    import doctest
    doctest.testmod()

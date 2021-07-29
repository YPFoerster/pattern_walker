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
import pattern_walker.utils as utils

__all__ = [
    'walker', 'patternWalker', 'fullProbPatternWalker', 'sectionedPatternWalker', 'SRPWalker', 'make_tree',
    'mutate_pattern'
    ]

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
        self.add_nodes_from(G.nodes(data=True))
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

    pattern_len-- Length of binary strings assined to all nodes.

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

    def set_target(self,node):
        """Set new target along with corresponding target pattern.
        """
        if node in self.nodes():
            self.target_node=node
            self.target_pattern=self.nodes[node]['pattern']
        else:
            print("Can't set {} as target. Node is not in graph.".format(node))

    def set_patterns(self):
        """
        Assigns a binary string to every node in the graph, successively by
        generation/level. A pattern/string is generated based on its parent
        string by the function mutate_pattern.
        """
        self.nodes[self.root]['pattern']=np.random.randint(
                                                        0,2,self.pattern_len
                                                        )
        last_patterned_gen=[self.root]
        queue=list(self.successors(self.root))
        while len(queue)>0:
            for node in queue:
                pattern=self.nodes[list(self.hierarchy_backup.predecessors(node))[0]]['pattern']
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
        self.add_nodes_from(self.hierarchy_backup.nodes(data=True))
        self.add_edges_from(self.hierarchy_backup.edges())
        self.set_patterns()
        self.set_target(self.target_node)
        self.set_weights()

    def num_pattern_duplicates(self):
        """Count all strings that appear more than once."""
        patterns = list(nx.get_node_attributes(self,'pattern').values())
        duplicates = []
        uniques = set([ pattern.tostring() for pattern in patterns ] )
        return len(patterns)-len(uniques)

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

class fullProbPatternWalker(patternWalker):

    def __init__(self,G,init_pos,pattern_len,root_prior,low_child_prior,high_child_prior,overlap,flip_rate,root_flip_rate,metric=None,search_for=None):
        target=None
        if search_for is None:
            #In case a seed is fixed, this needs to be done first, otherwise
            #otherwise the target changes with the overlap.
            target=np.random.choice(utils.leaves(G))
        elif search_for in G.nodes:
            target=search_for
        self.root_prior=root_prior
        self.high_child_prior=high_child_prior
        self.low_child_prior=low_child_prior
        self.overlap=overlap
        self.root_flip_rate=root_flip_rate
        self.num_sections=self.set_position_numbers(G,init_pos)
        self.sec_size=int(pattern_len/self.num_sections)
        super(fullProbPatternWalker,self).__init__(G,init_pos,pattern_len,flip_rate,metric,target)

    def set_position_numbers(self,G,init_pos):
        G.nodes[init_pos]['section']=0 #descendant from which child of root?
        G.nodes[init_pos]['depth']=0 #distance from root
        section_heads=G.successors(init_pos)
        sec_counter=1
        for head in section_heads:
            G.nodes[head]['section']=sec_counter
            G.nodes[head]['depth']=1
            for descendant in nx.descendants(G,head):
                G.nodes[descendant]['section']=sec_counter
                G.nodes[descendant]['depth']=nx.shortest_path_length(G,init_pos,descendant)
            sec_counter+=1
        return sec_counter-1

    def set_patterns(self):
        """
        Assigns a binary string to every node in the graph, successively by
        generation/level. A pattern/string is generated based on its parent
        string by the function mutate_pattern.
        """
        self.nodes[self.root]['pattern']=np.random.choice([0,1],p=[1-self.root_prior,self.root_prior],
            replace=True,size=self.pattern_len
                                            )
        last_patterned_gen=[self.root]
        queue=list(self.successors(self.root))

        #Treat branch heads separately, due to different flip rate
        for head in queue:
            sec_ndx=self.nodes[head]['section']
            left_sec_bound=(sec_ndx-1)*self.sec_size-self.overlap
            in_sec = np.array(range(0,min(self.sec_size+2*self.overlap,self.pattern_len)))
            out_sec = np.array(range(len(in_sec),self.pattern_len))
            pattern=np.roll(self.nodes[self.root]['pattern'],-left_sec_bound)
            pattern[in_sec]=mutate_pattern(
                                        pattern[in_sec],self.root_flip_rate,self.root_prior,self.high_child_prior
                                            )
            if len(out_sec):
                pattern[out_sec]=mutate_pattern(
                                            pattern[out_sec],self.root_flip_rate,self.root_prior,self.low_child_prior
                                                )
            self.nodes[head]['pattern']=np.roll(pattern,left_sec_bound)

        queue=[suc for node in queue for suc in self.successors(node)]

        while len(queue)>0:
            for node in queue:
                sec_ndx=self.nodes[node]['section']
                left_sec_bound=(sec_ndx-1)*self.sec_size-self.overlap
                in_sec = np.array(range(0,min(self.sec_size+2*self.overlap,self.pattern_len)))
                out_sec = np.array(range(len(in_sec),self.pattern_len))
                pattern=np.roll(self.nodes[list(self.hierarchy_backup.predecessors(node))[0]]['pattern'],-left_sec_bound)
                pattern[in_sec]=mutate_pattern(
                                            pattern[in_sec],self.flip_rate,self.high_child_prior,self.high_child_prior
                                                )
                if len(out_sec):
                    pattern[out_sec]=mutate_pattern(
                                                pattern[out_sec],self.flip_rate,self.low_child_prior,self.low_child_prior
                                                    )
                self.nodes[node]['pattern']=np.roll(pattern,left_sec_bound)
            queue=[suc for node in queue for suc in self.successors(node)]

class sectionedPatternWalker(patternWalker):
    """
    Example:
    >>> from networkx.generators.classic import balanced_tree
    >>> G=balanced_tree(1,10,create_using=nx.DiGraph)
    >>> root=0
    >>> w=sectionedPatternWalker(G,root,pattern_len=20,flip_rate=0.005,sections=0.5)
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
        target=None
        if search_for is None:
            #In case a seed is fixed, this needs to be done first, otherwise
            #otherwise the target changes with the overlap.
            target=np.random.choice(utils.leaves(G))
        elif search_for in G.nodes:
            target=search_for

        self.sections=self.sections_prep(G,init_pos,pattern_len,sections)
        self.num_sections=len(self.sections)
        full_pattern_len=self.sections[-1][-1]-self.sections[0][0]
        super(sectionedPatternWalker,self).__init__(G,init_pos,full_pattern_len,flip_rate,metric,target)


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

        main_brances=list(self.successors(self.root))
        for i in range(self.num_sections):
            self.nodes[main_brances[i]]['pattern']=[ 0 if ind < self.sections[i][0] or ind>=self.sections[i][1] else self.nodes[main_brances[i]]['pattern'][ind] for ind in range(self.pattern_len) ]
            #self.nodes[main_brances[i]]['pattern'][:section_boundaries[i]]=[0]*section_boundaries[i]
            #self.nodes[main_brances[i]]['pattern'][section_boundaries[i+1]:section_boundaries[-1]]=[0]*(section_boundaries[i+1]-section_boundaries[-1])

            for node in nx.descendants(self,main_brances[i]):
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
        elif isinstance(sections,float):
            out=sections_by_overlap(pattern_len,len(list(G.successors(init_pos))),sections)
        return out


class SRPWalker(sectionedPatternWalker):
    """
    Example:
    >>> from networkx.generators.classic import balanced_tree
    >>> G=balanced_tree(1,10,create_using=nx.DiGraph)
    >>> root=0
    >>> w=SRPWalker(G,root,pattern_len=20,flip_rate=0.005,sections=0.5,num_refs=2)
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
    >>> w.reset_patterns()
    >>> w.references[0] in w.edges()
    True
    """
    def __init__(self,G,init_pos,pattern_len,flip_rate,sections,num_refs,metric=None,search_for=None):
        target=None
        if search_for is None:
            #In case a seed is fixed, this needs to be done first, otherwise
            #otherwise the target changes with the overlap.
            target=np.random.choice(utils.leaves(G))
        elif search_for in G.nodes:
            target=search_for
        self.references=[np.random.choice(G.nodes(),2) for _ in range(num_refs)]
        super(SRPWalker,self).__init__(G,init_pos,pattern_len,flip_rate,sections,metric,target)
        self.add_edges_from(self.references)

    def reset_patterns(self):
        """
        Like set_patterns, but first resets edge data, because set_weights
        introduces edges blurring the hierarchy on that set_patterns relies.
        """
        self.clear()
        self.add_edges_from(self.hierarchy_backup.edges())
        self.set_patterns()
        self.set_target(self.target_node)
        self.add_edges_from(self.references)
        self.set_weights()


def hamming_dist(a,b):
    """Return number of non-equal entries of a and b."""
    return np.linalg.norm(a-b,ord=1)

def mutate_pattern(pattern,gamma,parent_prior=0.5,child_prior=None):
    """Expect a binary string and flip every entry with probability gamma,
    modified by the marginal expectation of each bit."""
    if child_prior is None:
        child_prior=parent_prior
    flip_prob=flip_probability_handle(child_prior*gamma,parent_prior,child_prior)
    pattern=list(pattern)
    return np.array([ 1-x if np.random.random()<flip_prob(x) else x for x in pattern ])

def flip_probability_handle(gamma,parent_prior,child_prior):
    """Returns probabilty function to flip a bit depending on its state and
    marginal expectations of parent and child."""
    # if root_prior==0.5:
    #     return lambda state: gamma
    if parent_prior==0:
        return lambda state: gamma
    else:
        def out_func(state):
            if state:
                return 1-(child_prior-(1-parent_prior)*gamma)/parent_prior
            else:
                return gamma
        return out_func

def sections_by_overlap(pattern_len,num_sections,frac_overlap):
    #TODO Test this
    overlap=int(frac_overlap*pattern_len)
    sections=[ ( i*(pattern_len-overlap),i*(pattern_len-overlap)+pattern_len ) for i in range(num_sections)]
    return sections

def make_tree(lam,pattern_len,flip_rate,overlap,n_max=100,seed=None):
    #TODO Test this
    def maker():
        H,root=utils.poisson_ditree(lam,n_max)
        target=np.random.choice(utils.leaves(H))
        G=sectionedPatternWalker(H.copy(),root,pattern_len,flip_rate,overlap,search_for=target)
        G.set_weights()
        return H,root,G
    if isinstance(seed,int):
        print('Using seed {}'.format(seed))
        maker=utils.seed_decorator(maker,seed)
    return maker()


if __name__=="__main__":
    import doctest
    doctest.testmod()

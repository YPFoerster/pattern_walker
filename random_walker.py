"""
Asymmetric random walkers on a directed graph and some useful functions.

walker-- Basic biased random walker derived from nx.Digraph.
patternWalker-- Random walker derived from walker, searching a binary pattern.
count_pattern_duplicates-- Return number of duplicate patterns of a
    patternWalker.
"""

import numpy as np
import networkx as nx
import pattern_walker.utils as utils

__all__ = [
    'walker', 'patternWalker', 'fullProbPatternWalker', 'patternStats'
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

    def __init__(self,G,root,bias):
        """Initialise Digraph as G and instantiate class variables."""
        super(walker, self).__init__()
        self.add_nodes_from(G.nodes(data=True))
        self.add_edges_from(G.edges())
        self.bias=bias
        self.trace=[root]
        self.x=root
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

    def get_transition_matrix(self,target=None,nodelist=None) -> np.array:
        """
        Return transition matrix for nodes as given by nodelist. Either target
        or nodelist must be given.
        If no nodelist is given, take first row/column for the root and
        last row/column for the target; all other nodes appear in order of
        self.nodes(). If nodelist is given, taget kwargs is ignored.
        """
        if nodelist is None:
            if target is None:
                raise Exception('Either target or nodelist must be set.')
            nodelist=set(self.nodes)-set([self.root])-set([target])
            nodelist=[self.root]+list(nodelist)+[target]
        return nx.to_numpy_array(self,nodelist=nodelist,weight='weight')

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

    Gamma-- Probability to change any string bit from one node to its child.

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

    def __init__(self,G,root,pattern_len,Gamma,metric=None,target=None):
        """
        Initialise variables as described, passing G and it's root to the
        the superclass. Calls set_patterns to assign binary strings to all
        nodes.

        G-- Graph data, must be compatible with class walker.
        root-- inital postion of the walker. Will be handled as root of G
        pattern_len-- Length of binary strings assigned to nodes.
        Gamma-- Probability of changing any bit in the string propagated
            from parent to child node.
        metric-- Metric for binary strings. (if None: Hamming distance)
        target-- The target node of the walker. If None, one is chosen
            randonly from the leaf nodes.
        """
        #Remember to pass walker.bias=1 in super
        self.hierarchy_backup=G.copy()
        super(patternWalker,self).__init__(G,root,1.)
        self.pattern_len=pattern_len
        self.Gamma=Gamma
        self.root=root
        if metric is None:
            self.metric=utils.hamming_dist
        else:
            self.metric=metric
        self.set_patterns()
        if target is None:
            self.target_node=np.random.choice(utils.leaves(self.hierarchy_backup))
            self.target_pattern=self.nodes[self.target_node]['pattern']
        else:
            self.target_node=target
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
        mutation_probs=self.mutation_probs()
        last_patterned_gen=[self.root]
        queue=list(self.successors(self.root))
        while len(queue)>0:
            for node in queue:
                pattern=self.nodes[list(self.hierarchy_backup.predecessors(node))[0]]['pattern']
                self.nodes[node]['pattern']=self.mutate_pattern(
                                                pattern,mutation_probs
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

    def mutation_probs(self):
        out=[ self.Gamma,self.Gamma ]
        return out

    def mutate_pattern(self,pattern,mutation_probs):
        """Expect a binary string and flip every entry with probability gamma,
        modified by the marginal expectation of each bit."""
        pattern=list(pattern)
        return np.array([ 1-x if np.random.random()<mutation_probs[x] else x for x in pattern ])


class fullProbPatternWalker(patternWalker):
    """
    Evolution of the patternWalker-class. For the root-pattern every marginal
    expectation is the same ($a$ in manuscript). lower patterns have bits of
    high ($a_h$) and low ($a_l$) marginal expectation, the number of which
    is given by [pattern length]/[number of branches] + 2*Overlap ($L/c+2\Delta$)
    for the former and L less that number ($L(c-1)/c-2\Delta$) for the latter.
    Overlap ($\Delta$) controls the number of bits which on neihbouring branches
    both have high expectation $a_h$. Gamma_root ($\Gamma^\prime$) is the
    mutation rate from root- to part(i.e.first level)-patterns, Gamma ($\Gamma$)
    the mutation rate on lower levels.

    Overriding methods:
    get_probs-- Return transition probabilities from given node. Overrides
        patternWalker.get_probs.

    Extending methods:
    set_patterns-- Propagate binary strings from root to all nodes. Constructing
    intervals of "typical keywords" of size $L/c+2\Delta$ with approprate overlap
    between neighbours.

    reset_patterns-- call parent function, then set_coordinates

    Extending variables:
    a_root -- probability that a given bit of the root pattern is 1

    Gamma_root -- probability that a given bit of a Part pattern is
        is 1 given that the same bit for the root pattern is 0

    a_low -- probability that a given (Part-generic) bit is 1

    a_high -- probability that a given (Part-specific) bit is 1
    """

    def __init__(self,G,root,pattern_len,a_root,a_low,\
        a_high,overlap,Gamma,Gamma_root,metric=None,\
        target=None):
        if target is None:
            #In case a seed is fixed, this needs to be done first,
            #otherwise the target changes with the overlap.
            target=np.random.choice(utils.leaves(G))
        elif target in G.nodes:
            target=target
        self.a_root=a_root
        if a_high<=(1-a_root)*Gamma_root+a_root and a_high>=(1-a_root)*Gamma_root:
            self.a_high=a_high
        else:
            self.a_high=(1-a_root)*Gamma_root+a_root
        if a_low>=(1-a_root)*Gamma_root and a_low<=(1-a_root)*Gamma_root+a_root:
            self.a_low=a_low
        else:
            self.a_low=(1-a_root)*Gamma_root+a_root/10
        self.overlap=overlap
        self.Gamma_root=Gamma_root
        self.num_parts=self.set_position_numbers(G,root,target)
        self.part_size=int(pattern_len/self.num_parts)
        self.coordinates_set=False
        super(fullProbPatternWalker,self).__init__(G,root,pattern_len,Gamma,metric,target)

    def set_position_numbers(self,G,root,target):
        """
        Assign to each node its Part and depth for convenient access.
        """
        G.nodes[root]['Part']=0 #root has its own "zeroth" Part
        G.nodes[root]['depth']=0 #distance from root

        #we prefer to have the target in part numero 1
        target_part= nx.shortest_path(G,root,target)[1]

        G.nodes[target_part]['Part']=1
        G.nodes[target_part]['depth']=1
        #the Part level node defines the Part of all its descendants
        for descendant in nx.descendants(G,target_part):
            G.nodes[descendant]['Part']=1
            G.nodes[descendant]['depth']=nx.shortest_path_length(G,root,descendant)
        #remaining Parts in any order enumerated from 2
        other_parts=set(G.successors(root))-set([target_part])

        for ndx,part in enumerate(other_parts,start=2):
            G.nodes[part]['Part']=ndx
            G.nodes[part]['depth']=1
            for descendant in nx.descendants(G,part):
                G.nodes[descendant]['Part']=ndx
                G.nodes[descendant]['depth']=nx.shortest_path_length(G,root,descendant)
        #return number of parts to check all went well
        return len(other_parts)+1

    def set_coordinates(self):
        """
        5-vector labels for nodes, for position relative to target.
        """
        #height of the tree
        h=nx.shortest_path_length(self,self.root,self.target_node)
        target_part=self.nodes[self.target_node]['Part']
        for node in self.nodes:
            #in the target-Part, we don't need to cross the root to get to target
            if self.nodes[node]['Part']==target_part:
                self.nodes[node]['coordinates']=(nx.shortest_path_length(self,self.target_node,node),0,0,0,target_part-1)
            #from root, we need to cross one root-part edge to get to target-part
            elif node==self.root:
                self.nodes[node]['coordinates']=(h-1,1,0,0,0)
            #otherwise, we need to go over root
            else:
                self.nodes[node]['coordinates']=( h-1,1,1,nx.shortest_path_length(self,node,self.root)-1,self.nodes[node]['Part'] )
        #record that we have assigned coordinates
        self.coordinates_set=True

    def set_patterns(self):
        """
        Assigns a binary string to every node in the graph, successively by
        generation/level. A pattern/string is generated based on its parent
        string by the function mutate_pattern.
        """
        self.nodes[self.root]['pattern']=np.random.choice([0,1],p=[1-self.a_root,self.a_root],
            replace=True,size=self.pattern_len
                                            )
        last_patterned_gen=[self.root]
        queue=list(self.successors(self.root))
        mutation_probs_a_high=self.root_mutation_probs(self.a_high)
        mutation_probs_a_low=self.root_mutation_probs(self.a_low)
        #Treat Parts separately, due to different flip rate
        for head in queue:
            part_ndx=self.nodes[head]['Part']
            #define index ranges for (Part-)"specific" and "generic" bits

            #firstly, take parent pattern and roll so that the leftmost specific
            #bit is at 0 index 0
            left_part_bound=int((part_ndx-1)*self.part_size-self.overlap)
            pattern=np.roll(self.nodes[self.root]['pattern'],-left_part_bound)

            #define index sets for specific and generic bits
            print(self.part_size+2*self.overlap,self.pattern_len,left_part_bound)
            in_part = np.array(range(0,min(int(self.part_size+2*self.overlap),self.pattern_len)))
            out_part = np.array(range(len(in_part),self.pattern_len))

            #mutate (1) specific and (2) generic bits separately
            pattern[in_part]=self.mutate_pattern(
                                        pattern[in_part],mutation_probs_a_high
                                        )
            if len(out_part):
                pattern[out_part]=self.mutate_pattern(
                                            pattern[out_part],mutation_probs_a_low
                                            )
            #undo rolling to have bits in the right positions
            self.nodes[head]['pattern']=np.roll(pattern,left_part_bound)

        #prepare queue for next level of patterns: go over nodes currently in
        #the queue, make new queue with their successors
        queue=[successor for node in queue for successor in self.successors(node)]
        mutation_probs_a_high=self.part_mutation_probs(self.a_high)
        mutation_probs_a_low=self.part_mutation_probs(self.a_low)

        while len(queue)>0:
            for node in queue:
                part_ndx=self.nodes[node]['Part']
                #define index ranges for (Part-)"specific" and "generic" bits

                #firstly, take parent pattern and roll so that the leftmost specific
                #bit is at 0 index 0
                left_part_bound=int((part_ndx-1)*self.part_size-self.overlap)
                pattern=np.roll(
                    self.nodes[list(self.hierarchy_backup.predecessors(node))[0]]['pattern'],\
                    -left_part_bound
                    )

                #define index sets for specific and generic bits
                in_part = np.array(range(0,min(int(self.part_size+2*self.overlap),self.pattern_len)))
                out_part = np.array(range(len(in_part),self.pattern_len))

                #mutate (1) specific and (2) generic bits separately
                pattern[in_part]=self.mutate_pattern(
                                            pattern[in_part],mutation_probs_a_high
                                            )
                if len(out_part):
                    pattern[out_part]=self.mutate_pattern(
                                                pattern[out_part],mutation_probs_a_low
                                                    )
                # undo rollling
                self.nodes[node]['pattern']=np.roll(pattern,left_part_bound)
            #new queue as above
            queue=[suc for node in queue for suc in self.successors(node)]

    def gen_mutation_probs(self,a_parent,a_child,Gamma):
        if a_parent:
            out=[ Gamma,(1-(a_child-(1-a_parent)*Gamma)/a_parent) ]
        else:
            out=[ Gamma,0 ]
        return out

    def root_mutation_probs(self,a_child):
        if self.a_root:
            out=[self.Gamma_root,1-(a_child-(1-self.a_root)*self.Gamma_root)/self.a_root]
        else:
            out=[self.Gamma_root,0]
        return out

    def part_mutation_probs(self,a_child):
        if a_child:
            out=[a_child*self.Gamma,(1-a_child)*self.Gamma]
        else:
            out=[a_child*self.Gamma,0]
        return out

    def reset_patterns(self):
        #Be sure that coordinates are set again after calling nx.clear()
        patternWalker.reset_patterns(self)
        if self.coordinates_set:
            self.set_coordinates()

class patternStats(fullProbPatternWalker):
    """
    Wrapper class for statistical properties of patterns, analytical and simulated.
    """
    def __init__(self,G,root,pattern_len,a_root,a_low,\
        a_high,overlap,Gamma,Gamma_root,metric=None,\
        target=None):
        super(patternStats, self).__init__(
            G,root,pattern_len,a_root,a_low,a_high,\
                overlap,Gamma,Gamma_root,metric=None,target=None
                )
        #the following are handy to have at hand
        self.leaves=utils.leaves(G)
        self.parts=list(nx.neighbors(G,root))
        #leaves grouped by their parts
        self.part_leaves=[
            [node for node in  nx.descendants(G,part) if node in self.leaves]
            for part in self.parts
            ]
        #coordination number and height of tree
        self.c=len(list(self.successors(root)))
        self.h=nx.shortest_path_length(
            G,root,self.leaves[0]
            )
        #offset of marginal expectations from extreme values
        self.beta_l=self.a_low-(1-self.a_root)*self.Gamma_root
        #beta_h not used in paper any more (set ot 0), but result still valid
        self.beta_h=(1-self.a_root)*self.Gamma_root+self.a_root-\
            self.a_high

    def expected_part_dist(self):
        """
        Analytically expected pattern distance between neighbouring
        parts.
        """

        #to simplify expressions
        L=self.pattern_len
        c=self.c
        h=self.h
        a=self.a_root
        Gammap=self.Gamma_root
        Gamma=self.Gamma
        Delta=self.overlap
        beta_h=self.beta_h
        beta_l=self.beta_l

        if beta_h==0:
            if Delta<=L*(c-2)/(2*c):
                return 2*L*Gammap*(1-Gammap)*\
                    (1-a)+2*L/c*(a-beta_l)*((c-2)*beta_l/a+1)-\
                    4*Delta*(a-beta_l)*beta_l/a
            else:
                return 2*L*Gammap*(1-Gammap)*(1-a)+2*L/c*(c-1)*(a-beta_l)-\
                    4*Delta*(a-beta_l)

        else:
            if Delta<=L*(c-2)/(2*c):
                return 2*L*Gammap*(1-Gammap)*(1-a)+\
                    2*L/c*(c*beta_l*(a-beta_l)/a+a+beta_l-\
                    beta_h-2*beta_l*(2*a-beta_l-beta_h)/a)+\
                    4*Delta/a*((a-beta_h)*beta_h-(a-beta_l)*beta_l)
            else:
                return 2*L*Gammap*(1-Gammap)*(1-a)-\
                2*L/c*(a-beta_h)/a*( (beta_h+2*beta_l)*c-2*(beta_h+beta_l) )+\
                2*L/c*(c-1)*(a+beta_l-beta_h)+\
                4*Delta*(2/a*(a-beta_h)*(beta_h+beta_l)-a-beta_l+beta_h)


    def root_to_part_rate(self,a_j,Gammap):
        """
        Probability that a given bit of the root pattern is different from
        the same bit in a Part pattern
        """
        a=self.a_root
        return 2*(1-a)*Gammap+a-a_j

    def expected_root_to_part_distance(self):
        """
        Expected distance between root pattern and any of the part patterns
        """
        L=self.pattern_len
        c=self.c
        h=self.h
        a=self.a_root
        Gammap=self.Gamma_root
        Gamma=self.Gamma
        Delta=self.overlap
        beta_h=self.beta_h
        beta_l=self.beta_l

        a_l=(1-a)*Gammap+beta_l
        a_h=(1-a)*Gammap+a
        return self.root_to_part_rate(a_h,Gammap)*(L/c+2*Delta)+\
            self.root_to_part_rate(a_l,Gammap)*(L-L/c-2*Delta)


    def vertical_rate(self,a_j,Gamma,h):
        """
        Given a pattern (not the root) and an h-th degree descendant, return
        probability that a given bit is differnt between the patterns
        """
        return 2*a_j*(1-a_j)*(1-(1-Gamma)**float(h-1))

    def expected_vertical_distance(self):
        """
        Analytically expected pattern distance between a part-pattern and a
        leaf pattern of the same part.
        """
        L=self.pattern_len
        c=self.c
        h=self.h
        a=self.a_root
        Gammap=self.Gamma_root
        Gamma=self.Gamma
        Delta=self.overlap
        beta_h=self.beta_h
        beta_l=self.beta_l

        a_l=(1-a)*Gammap+beta_l
        a_h=(1-a)*Gammap+a
        return self.vertical_rate(a_h,Gamma,h)*(L/c+2*Delta)+\
            self.vertical_rate(a_l,Gamma,h)*(L-L/c-2*Delta)

    def sample_distances(self,number_of_samples):
        """
        Generate lots of sets patterns, and calculate the average pattern-
        distance between
        (1) neighbouring parts
        (2) parts and their leaf-descendants
        (3) root and the parts
        """
        part_mean_distances=np.zeros(number_of_samples)
        vertical_mean_distances=np.zeros(number_of_samples)
        root_part_mean_distances=np.zeros(number_of_samples)
        for iter in range(number_of_samples):
            part_mean_distances[iter]=np.mean([
                    np.linalg.norm(self.nodes[self.parts[n1]]['pattern']-\
                    self.nodes[self.parts[n1+1]]['pattern'],ord=1)
                for n1 in range(len(self.parts)-1)
                ] )
            vertical_mean_dist=np.mean( [
                    [
                        np.linalg.norm(self.nodes[self.parts[part_ndx]]['pattern']-\
                        self.nodes[leaf]['pattern'],ord=1)
                    for leaf in self.part_leaves[part_ndx]
                    ]
                for part_ndx in range(self.c)
                ] )
            vertical_mean_distances[iter]=vertical_mean_dist
            root_part_mean_distances[iter]=np.mean([
                    np.linalg.norm( self.nodes[self.parts[part_ndx]]['pattern']-\
                    self.nodes[self.root]['pattern'],ord=1)
                for part_ndx in range(self.c)
                ])
            self.reset_patterns()
        return part_mean_distances,vertical_mean_distances,root_part_mean_distances


if __name__=="__main__":
    import doctest
    doctest.testmod()

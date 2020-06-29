import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from dag_exps import poisson_ditree

class walker:
    def __init__(self,G,bias,init_pos):
        self.G=G
        self.bias=bias
        self.trace=[init_pos]
        self.x=init_pos
        self.t=0

    def get_probs(self,site):
        children=list(self.G.successors(site))
        parents=list(self.G.predecessors(site))
        #print('parents:',parents)
        #print('children:',children)
        #probs=[]
        #probs=[1/(self.bias*len(parent)+len(children))]*len(children)+[self.bias/(self.bias*len(parent)+len(children))]*len(parent)
        probs={}
        for child in children:
            probs[child]=1./(self.bias*len(parents)+len(children))
        for parent in parents:
            probs[parent]=self.bias/(self.bias*len(parents)+len(children))
        #print( probs.values() )
        return children,parents,probs


    def set_weights(self):
        for site in self.G.nodes:
            probs={}
            parents=[]
            children=[]
            children,parents,probs=self.get_probs(site=site)
            nx.set_edge_attributes(self.G,{(site,child):{'prob':probs[child]} for child in children})
            self.G.add_edges_from( [ (site,parent,{'prob': probs[parent]}) for parent in parents ] )


    def step(self):
        out_edges_list=self.G.out_edges(self.x,'prob')
        self.x = np.random.choice( [x[1] for x in out_edges_list],p=[x[2] for x  in out_edges_list] )
        self.trace.append(self.x)
        self.t+=1

    def reset(self):
        self.x=self.trace[0]
        self.trace=[self.root]
        self.t=0

def hamming_dist(a,b):
    #return np.count_nonzero(np.array(a)!=np.array(b))
    temp=[a[i]-b[i] for i in range(len(a))]
    return np.count_nonzero(temp)

class patternWalker(walker):
    def __init__(self,G,init_pos,pattern_len,flip_rate,metric=hamming_dist,search_for=None):
        # NOTE: It should be checked that "G" is directed and rooted in init_pos.
        super(patternWalker,self).__init__(G,1.,init_pos)
        self.pattern_len=pattern_len
        self.flip_rate=flip_rate
        self.root=init_pos
        self.metric=metric
        self.G.nodes[self.root]['pattern']=list(np.random.randint(0,2,self.pattern_len))
        self.set_patterns()
        if search_for is None:
            self.searched_node=np.random.choice(self.G.nodes)
            self.searched_pattern=self.G.nodes[self.searched_node]['pattern']
        else:
            self.searched_node=search_for
            self.searched_pattern=self.G.nodes[self.searched_node]['pattern']


    def set_patterns(self):
        last_patterned_gen=[self.root]
        queue=list(self.G.successors(self.root))
        while len(queue)>0:
            for node in queue:
                pattern=self.G.nodes[list(self.G.predecessors(node))[0]]['pattern']
                self.G.nodes[node]['pattern']=mutate_pattern( pattern,self.flip_rate )
            #temp=[]
            #for node in queue:
            #    for suc in self.G.successors(node):
            #        temp.append(suc)
            #queue=temp
            queue=[suc for node in queue for suc in self.G.successors(node)]


    def get_probs(self,site):
        # TODO: One could inherit this method of "walker" and isolate the actual
        # probability assignments
        children=list(self.G.successors(site))
        parents=list(self.G.predecessors(site))
        probs={}
        for child in children:
            probs[child]=1/(1+self.metric( self.G.nodes[child]['pattern'],\
                self.searched_pattern ))
        for parent in parents:
            probs[parent]=1/(1+self.metric( self.G.nodes[parent]['pattern'],\
                self.searched_pattern ))
        denominator = sum( probs.values() )
        for key in probs.keys():
            probs[key]/=denominator

        return children,parents,probs

    def get_mfpt(self):
        trans = nx.to_numpy_matrix(self.G,weight='prob').T
        target_index=list(self.G.nodes()).index(self.searched_node)
        root_index=list(self.G.nodes()).index(self.root)
        evals,r_evecs=np.linalg.eig(trans)
        r_evecs=np.array(r_evecs)
        max_eval_index=np.argmax(evals)
        l_evecs=np.linalg.inv(r_evecs)
        eq_evec=l_evecs[ max_eval_index  ]
        out = 1/eq_evec[root_index]*( 1+ np.sum([ evals[l]/(1-evals[l])*r_evecs[root_index,l]*( l_evecs[l,root_index]-l_evecs[l,target_index]) for l in range(len(evals)) if l != max_eval_index]) )
        return out

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


    G,root=poisson_ditree(2,5000)
    H,_=poisson_ditree(2,5000)
    #G.add_edges_from(H)
    while len(G)==2:
        print("Too short tree, trying again...")
        G,root=poisson_ditree(2,5000)
    connection_points_G=np.random.choice(list(G.nodes),100)
    connection_points_H=np.random.choice(list(H.nodes),100)
    G.add_edges_from(H.edges)
    G.add_edges_from([ (connection_points_H[i],connection_points_G[i]) for i in range(len(connection_points_G))])


    print("Tree generated.")
    rw=walker(G,0.5,root)
    for iter in range(1000):
        rw.step()
    frequency={}
    for item in rw.trace:
        frequency[item]=rw.trace.count(item)
    #print(frequency)
    pos=nx.spring_layout(G)
    nx.draw(G,node_color='blue')
    nx.draw(G, nodelist=list(frequency.keys()), node_color=list(frequency.values()),cmap=plt.cm.get_cmap('Reds'))
    plt.show()

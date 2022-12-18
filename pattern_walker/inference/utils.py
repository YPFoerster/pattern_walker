# data and text processing
import json
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer,TfidfVectorizer
import spacy


#technical packages
import numpy as np
from scipy.stats import binom,poisson,chisquare,linregress
import networkx as nx
from itertools import count

#visualisation
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout

#handy
import pickle
import sys



def load_data(file):
    with open (file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return (data)

def write_data(file, data):
    with open (file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def flat_json(data,string='name',verbose=False):
    out=[]
    queue=[]
    out.append(data[string])
    queue.extend(data['_children'])
    while len(queue)>0:
        temp=queue.pop(-1)
        out.append(temp[string])#not checked if already seen
        try:
            queue.extend(temp['_children'])
        except KeyError as e:
            if verbose:
                print('Found a leaf: No attribute {}'.format(e))

    return out

def flat_parts_json(data,**kwargs):
    parts = [flat_json(part_data,**kwargs) for part_data in data['_children'] ]
    return parts

def get_node_id(node_dict,gen_node):
    #check if dict has entry for 'node_id', and if not, make one
    try:
        return node_dict['node_id']
    except KeyError:
        node_dict['node_id'] = gen_node()
        return node_dict['node_id']

def build_tree(data,string='name',int_ids=False,masks=True):
    if int_ids:
        counter=count(0)
        def gen_node():
            return next(counter)
    else:
        gen_node = nx.utils.generate_unique_node
    G=nx.DiGraph()

    root=get_node_id(data,gen_node)

    G.add_node(root,**{prop:data[prop] for prop in [string]})

    queue = []
    for part in data['_children']:

        node=get_node_id(part,gen_node)

        if masks:
            G.add_node(node,**{prop:part[prop] for prop in [string,'mask']})
        else:
            G.add_node(node,**{prop:part[prop] for prop in [string]})
        G.add_edge( root,node )

        queue.append(part)

    while len(queue)>0:
        parent=queue.pop(-1)
        try:
            children=parent['_children']
            for child in children:
                child_node=get_node_id(child,gen_node)

                G.add_node(child_node, **{ prop:child[prop] for prop in [string] } )
                G.add_edge( parent['node_id'],child_node )
            queue.extend(children)
        except KeyError as e:
            pass
    return G

def plot_tree(G,labelled=[],**label_kwargs):
    pos=graphviz_layout(G,prog='dot') #positions to display tree "nicely"
#     (edges,weights) = zip(*nx.get_edge_attributes(G,None).items())
    edges=G.edges()
    fig_handle=plt.figure(figsize=(40,20))
    nx.draw(G, pos, edgelist=edges, node_size=3000,arrowsize=100,width=5.0)
    try:
        nx.draw_networkx_nodes(G,pos,nodelist=labelled,node_color='r',\
                                    node_size=3000,node_shape='H',**label_kwargs)
        # nx.draw_networkx_labels( G,pos,labels= nx.get_node_attributes(G,'name') )
    except KeyError:
        pass

    return fig_handle,pos


def add_part_attribute(data,G):
    out = {}
    for part in data['_children']:
        part_node = part['node_id']
        out.update({ node:part_node for node in nx.descendants(G,part_node) })
        out.update({part_node:part_node})
    # out = { node:part for (node,part) in out }
    nx.set_node_attributes(G,out,'part')

def get_distances_to_parent(G,masked=True,nodes=None):
    out = {}
    if nodes is None:
        nodes = G.nodes

    for node in nodes:
        try:
            parent = list(G.predecessors(node))[0] #there is only one parent
            if masked:
                part_mask = G.nodes[node]['part']
                part_mask = np.array(G.nodes[part_mask]['mask'])
                temp = np.abs( (np.array(G.nodes[node]['name'])-np.array(G.nodes[parent]['name']))*part_mask  ).sum()
            elif masked=='inverse':
                part_mask = G.nodes[node]['part']
                part_mask = ~np.array(G.nodes[part_mask]['mask'])
                temp = np.abs( (np.array(G.nodes[node]['name'])-np.array(G.nodes[parent]['name']))*part_mask  ).sum()
            else:
                temp = np.abs( np.array(G.nodes[node]['name'])-np.array(G.nodes[parent]['name']) ).sum()
            out[node]=temp
        except IndexError:
            continue
    return out

def get_mask_length(G,nodes,masked=True):

    if nodes is None:
        nodes=[np.random.choice(G.nodes)]

    out = {node:len(G.nodes[node]['name'][0]) for node in nodes}
    for node in nodes:
        if masked==True or masked=='inverse':
            try:
                pt = G.nodes[node]['part']
            except KeyError:
                part_node =list(G.successors(node))[0]
                pt = G.nodes[part_node]['part']

            if masked==True:
                out[node] = np.sum(G.nodes[pt]['mask'])
            else:
                out[node] = out[node] - np.sum(G.nodes[pt]['mask'])

    return out

def get_masked_vectors(G,nodes,masked=True):

    # check if nodes is iterable, in which case we return a dircionary. if
    # single node, return just the vector
    # try:
    #     iter(nodes)
    # except TypeError:
    #     out_node = nodes
    #     nodes=[nodes]

    out = {}

    for node in nodes:
        if masked==True or masked=='inverse':
            try:
                pt = G.nodes[node]['part']
            except KeyError as e:
                continue

            if masked==True:
                vec = np.array(G.nodes[node]['name'])
                mask = np.array(G.nodes[pt]['mask'] )
                out[node]= vec*mask
            else:
                vec = np.array(G.nodes[node]['name'])
                mask = np.logical_not(G.nodes[pt]['mask'] )
                out[node]= vec*mask
        else:
            vec = np.array(G.nodes[node]['name'])
            out[node]= vec

    return out

def rel_flips_off_to_on(v1,v2):
    mask = np.array(v1)
    mask = np.where(mask==0,True,False)
    if np.sum(mask) == 0:
        return 0
    else:
        out = np.array(v2)
        out = np.sum( out[mask] )
    return out/np.sum(mask)

def rel_flips_on_to_off(v1,v2):

    mask = np.array(v1)
    mask = np.where(mask==1,True,False)
    if np.sum(mask) == 0:
        return 0
    else:
        out = np.array(v2)
        out = np.sum(mask)-np.sum( out[mask] )
    return out/np.sum(mask)


def estimate_a(G,nodes,masked=False):
    mask_len_dict = get_mask_length(G,nodes,masked)
    vectors = get_masked_vectors(G,nodes,masked)
    a = { node:np.sum(vector)/mask_len_dict[node] for (node,vector) in vectors.items() }
    return np.mean(list(a.values())),a

def estimate_Gamma(G,nodes,masked=True,fit_exp=True):
    Gamma = []
    for node in nodes:
        v1 = get_masked_vectors(G,[node],masked)[node]
        for child in G.successors(node):
            v2 = get_masked_vectors(G,[child],masked)[child]
            Gamma.append(rel_flips_off_to_on(v1,v2))

    out = {'Gamma':(np.mean(Gamma),Gamma)}
    if fit_exp:
        fit = fit_exponential(Gamma)
        out['fit']=fit
    return out

    return np.mean(Gamma),Gamma

def estimate_a_Gamma(G,nodes,masked=True,fit_exp=True):
    Gamma_01 = []
    Gamma_10 = []
    a = []
    mask_length_dict = get_mask_length(G,nodes,masked)
    for node in nodes:
        n = mask_length_dict[node]
        v1 = get_masked_vectors(G,[node],masked)[node]
        a.append( a_parent:=np.sum(v1)/n )
        for child in G.successors(node):
            m = mask_length_dict[child]
            v2 = get_masked_vectors(G,[child],masked)[child]
            a_child = np.sum(v2)/m
            Gamma_01.append(rel_flips_off_to_on(v1,v2))
            if a_parent>0:
                on_to_off = (rel_flips_on_to_off(v1,v2)-1+a_child/a_parent)*a_parent/(1-a_parent)
            else: on_to_off=0.
            Gamma_10.append(on_to_off)

    out = {'Gamma':(np.mean(Gamma_01+Gamma_10),Gamma_01,Gamma_10),'a':(np.mean(a),a)}
    if fit_exp:
        fit = fit_exponential(Gamma_01+Gamma_10)
        out['fit']=fit
    return out


def fit_exponential(data,**kwargs):
    counts,bins = np.histogram(data,density=True,**kwargs)

    mask=np.where(counts>0,True,False)

    fit=linregress( bins[:-1][mask],np.log(counts[mask]) )
    slope,intercept =fit.slope,fit.intercept
    return slope,intercept

def get_distance_histogram(G,masked=True,nodes=None,test_binom=True):

    fig,ax=plt.subplots()
    distances = get_distances_to_parent(G,masked=masked,nodes=nodes)
    max_dist = int(max(distances.values()))
    plot_dom = np.arange(0,max_dist+1.1,1.)

    mask_len_dict = get_mask_length(G,nodes,masked)
    n = list(mask_len_dict.values())[0] #best if all nodes are in the same part
    p = np.mean( [ val/n for val in distances.values() ] )

    counts,bins,_=ax.hist(distances.values(),bins=plot_dom,density=False,cumulative=False)
    ax.set_title('Masked: {}, p={}, n={}'.format(masked,p,n))

    if test_binom:
        rv = binom(n,p)
        f=lambda x: len(distances)*rv.pmf(x)
        ax.plot(plot_dom,f(plot_dom),'x')

        #determine the domain over which to compare the pmfs
        full_dom = list(np.arange(0,n+1,1)) #total possible domain
        max_arg_f = max(max_dist,full_dom[np.argmin( [ f(x) for x in full_dom ] )]) #domain over which we can divide by f (in chi^2 test)
        full_dom = np.arange(0,max_arg_f,1) # truncated domain
        full_counts = [ counts[i] if i <= max_dist else 0. for i in full_dom ] # append appropriate #0s to counts
        # full_counts = list(counts)+[0 for _ in range(max_arg_f-max_dist)] # append appropriate #0s to counts
        # full_counts = counts[:max_arg_f] #sometimes we might have move counts than relevant values of f?!
        # print(max_dist,len(full_counts),max_arg_f)
        print(chisq:=chisquare(full_counts,f(full_dom),ddof=1))

        return fig,ax,p,n,chisq

    else: return fig,ax,p,n

    # max_obs_dist = max(distances)
    # bins = np.arange(0.,max_obs_dist+1.1,1.)
    #
    # n = get_mask_length(G,masked)
    # p = np.mean(distances)/n
    #
    # counts,bins,_=ax.hist(distances,bins=bins,density=False,cumulative=False)
    # ax.set_title('Masked: '+str(masked)+" p={}, n={}".format(p,n))
    #
    # if test_binom:
    #     rv = binom(n,p)
    #     f=lambda x: len(distances)*rv.pmf(x)
    #     ax.plot(bins,f(bins),'x')
    #
    #     #determine the domain over which to compare the pmfs
    #     full_dom = np.arange(0,n+1,1) #total possible domain
    #     max_arg = np.argmin( [ f(x) if f(x)>1e-9 else 0. for x in full_dom] ) #domain over which counts and f are not 0 (useful in chi^2 test)
    #     full_dom = np.arange(0,max_arg+1,1) # truncated domain
    #     # full_counts = list(counts)+[0 for _ in range(len(full_dom)-max_obs_dist-1)]
    #     full_counts = [ counts[i] if i in range(len(counts)) else 0. for i in full_dom ] # append appropriate #0s to counts
    #
    #     print(chisquare(full_counts,f(full_dom),ddof=1))
    #
    # return fig,ax,full_counts,f(full_dom),p,n

def change_text_key(G,old_key,new_key, cast_to_array=True):
    for u, data in G.nodes(data=True):
        data[new_key] = np.array(data.pop(old_key)).squeeze()

act_stopwords = {'act','section','part','provision','subsection',
            'paragraph','chapter','acts',
            'sections','subsections','paragraphs','chapters' ,'provisions'
            }

nlp_default=spacy.load("en_core_web_sm", disable=["parser", "ner"])
nlp_default.Defaults.stop_words.update( act_stopwords )


def doc_lemmatizer(s,nlp=nlp_default,\
    allowed_postags=["NOUN","VERB","ADJ","ADV","PROPN"]):
    texts_out = []
    doc = nlp(s)
    lemmatized = []
    for token in doc:
        if token.pos_ in allowed_postags and not token.is_stop:
            lemmatized.append(token.lemma_)
    return lemmatized


def lemmatize_json(text_json,string='name',verbose=False,lemmatizer=doc_lemmatizer):
    queue=[]
    text_json_lem = text_json.copy()
    text_json_lem[string] = lemmatizer(text_json_lem[string])
    queue.extend(text_json_lem['_children'])

    while len(queue)>0:
        temp=queue.pop(-1)
        if type(temp[string])=="<class 'list'>":
            continue
        else:
            temp[string] = lemmatizer(temp[string])
            try:
                queue.extend(temp['_children'])
            except KeyError as e:
                if verbose:
                    print('Found a leaf: No attribute {}'.format(e))
    return text_json_lem

def vectorise_json(vectoriser,json,string='name',verbose = False,**kwargs):
    queue=[]
    out_json = json.copy()

    #this doesn't currently work to my satisfaction. best to pass a vectoriser
    if isinstance(vectoriser,list):
    #     if 'use_idf' in kwargs:
            # vectoriser = TfidfVectorizer(vocabulary=vectoriser,**kwargs)
    #     else:
            vectoriser = CountVectorizer(vocabulary=vectoriser,**kwargs)
    out_json[string] = vectoriser.fit_transform([out_json[string]]).toarray().tolist()
    if verbose:
        print(out_json[string])
    queue.extend(out_json['_children'])

    while len(queue)>0:
        node=queue.pop(-1)
        if isinstance(node[string],list):
            continue
        else:
            node[string] = vectoriser.transform([node[string]]).toarray().tolist()
            if verbose:
                print(node[string])
            try:
                queue.extend(node['_children'])
            except KeyError as e:
                if verbose:
                    print('Found a leaf: No attribute {}'.format(e))
    return out_json


def get_most_freq_words(data,num_top_terms=100,tokenizer=doc_lemmatizer,analyzer='words',vectoriser=None,**kwargs):

    if vectoriser is None:
        vectoriser = TfidfVectorizer(tokenizer=tokenizer,**kwargs)
    counts = vectoriser.fit_transform( data )
    cum_counts = np.array(counts.sum(axis=0)).squeeze()
    most_freq = np.argsort( cum_counts,axis=-1 )[-num_top_terms:]
    new_voc = [  ]
    old_voc = list(vectoriser.vocabulary_.keys())
    for key in most_freq:
        new_voc.append(old_voc[key])
    return new_voc, counts[:,-num_top_terms:]

def make_part_mask(full_vocab,part_vocab):
    temp = [ full_vocab.index(word) for word in part_vocab ]
    return [ 1 if i in temp else 0 for i in range(len(full_vocab))]

def get_vocabs(parts_flat,ranker=get_most_freq_words,**kwargs):
    lemmatizer = doc_lemmatizer
    part_vocab = [ranker(part,tokenizer=lemmatizer,**kwargs)[0] for part in parts_flat]

    full_vocab = []
    for ls in part_vocab:
        full_vocab.extend(ls)
    full_vocab = list(set(full_vocab))

    part_masks = [ make_part_mask(full_vocab,part) for part in part_vocab ]
    return full_vocab, part_vocab, part_masks

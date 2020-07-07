import random_walker as rw
from dag_exps import poisson_ditree,directify,leaves
import networkx as nx
from networkx.generators import balanced_tree
import numpy as np
import matplotlib.pyplot as plt

r=3 #offspring number
h=5 #height
N=100 #bits in a pattern; must be adapted to ensure uniqueness of patterns
gamma=0/N #mutation rate
number_of_samples=5000
max_time=5000

name_string='r{r}h{h}gamma{gamma}N{N}'.format(r=r,h=h,gamma=str(round(gamma,3)).replace('.','-'),N=N,number_of_samples=number_of_samples, max_time=max_time)
G=balanced_tree(r,h)
root=None
#root is the inly node in G with degree r, all others have degree r+1.
for node in G.nodes():
    if nx.degree(G,node)==r:
        root=node
        break
G=directify(G,root)[0]
target = leaves(G)[0]
walker=rw.patternWalker(G.copy(),root,N,gamma,search_for=target)
walker.set_weights()


#pi=walker.get_stat_distr()
#print(np.squeeze(pi)[0])
#P=nx.to_numpy_matrix(walker.G,weight='prob')
#print(P)
#balances=[ pi[x]*P[x,y]-pi[y]*P[y,x] for x in range(len(walker.G)) for y in range(len(walker.G)) if P[x,y] != 0]
#plt.hist(balances)
#plt.show()

YZ=walker.get_YZ_hitting_time()
AA=walker.get_mfpt()

fig,ax=plt.subplots(figsize=(14,8))
plt.hist(YZ,bins=50,alpha=0.5,label='laplacian',density=True)
plt.hist(AA,bins=50,alpha=0.5,label='eigen',density=True)
plt.xlabel('MFPT')
plt.ylabel('Frequency')
plt.legend(loc='best')
plt.savefig('./outputs/mfpts_comparision'+name_string+'.pdf')
plt.savefig('./outputs/mfpts_comparision'+name_string+'.png')
#plt.show()

#stat_distr  =walker.get_stat_distr()



"""

evals=np.array(walker.get_eval_distribution())
fig,ax=plt.subplots()
ax.hist(np.real(evals),color='b',alpha=0.3,bins=100)
#ax.hist(np.imag(evals),color='r',alpha=1)
plt.show()
"""
"""
print(nx.to_numpy_matrix(walker.G,weight='prob'))
print(len(walker.G),np.linalg.matrix_rank( nx.to_numpy_matrix(walker.G,weight='prob')))
mfpt=walker.get_mfpt()
print(mfpt)
"""

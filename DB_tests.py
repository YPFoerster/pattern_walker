import random_walker as rw
from dag_exps import poisson_ditree,directify,leaves
import networkx as nx
from networkx.generators import balanced_tree
import numpy as np
import matplotlib.pyplot as plt

r=3 #offspring number
h=5 #height
gamma=.0 #mutation rate
N=100 #bits in a pattern; must be adapted to ensure uniqueness of patterns
number_of_samples=2000
max_time=5000

name_string='r{r}h{h}N{N}'.format(r=r,h=h,N=N,number_of_samples=number_of_samples)

gamma_iter=np.arange(0.0,0.5,0.05)
balance_mean=[]
balance_std=[]


G=balanced_tree(r,h)
root=None
#root is the inly node in G with degree r, all others have degree r+1.
for node in G.nodes():
    if nx.degree(G,node)==r:
        root=node
        break

G=directify(G,root)[0]
target = leaves(G)[0]

for gamma in gamma_iter:
    balances=[]

    for iter in range(number_of_samples):
        walker=rw.patternWalker(G.copy(),root,N,gamma,search_for=target)
        walker.set_weights()

        pi=walker.get_stat_distr()
        P=nx.to_numpy_matrix(walker.G,weight='prob')
        balances.append([ np.abs(pi[x]*P[x,y]-pi[y]*P[y,x]) for x in range(len(walker.G)) for y in range(len(walker.G)) if P[x,y] != 0])
    balance_mean.append(np.mean(balances))
    balance_std.append(np.std(balances))

balance_mean-np.array(balance_mean)
balance_std=np.array(balance_std)

plt.plot(gamma_iter,balance_mean)
plt.plot(gamma_iter, balance_mean+2*balance_std,'k--')
plt.plot(gamma_iter, balance_mean-2*balance_std,'k--')
plt.xlabel('$\Gamma$')
plt.ylabel('$| \pi_{x}P_{xy}-\pi_{y}P_{yx}|$')
plt.title(name_string)
plt.savefig('DB_test.pdf')
plt.savefig('DB_test.png')
plt.show()
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

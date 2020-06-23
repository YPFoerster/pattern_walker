import networkx as nx
from networkx.utils import generate_unique_node
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def reg_ditree(n,k):
    """
    Returns a regular directed tree with 'n' generations and 'k' offspring nodes
    for each node that is not a leaf.
    """
    out=nx.DiGraph()
    current_gen=[generate_unique_node()]
    for gen in range(1,n):
        for node in current_gen:
            offspring = np.random.randint(1,k)
            new_gen=[generate_unique_node() for x in range(offspring)]
            out.add_edges_from( [ (node, child) for child in new_gen] )
            current_gen=new_gen.copy()
    return out



G=reg_ditree(100,10)

df=pd.DataFrame(columns=['ancestors','descendants'])
#df.columns=['ancestors','descendants']
for node in G:
    df.loc[node]=[len(nx.ancestors(G,node)),len(nx.descendants(G,node))]

some_node=list(G)[-1]
some_branch=list(nx.ancestors(G,some_node))+[some_node]+list(nx.descendants(G,some_node))
df_branch=df.loc[some_branch]

y=(df_branch.descendants-df_branch.ancestors)/(df_branch.descendants+df_branch.ancestors)
plt.scatter(x=df_branch['ancestors'],y=y)
plt.xlabel('Ancestors')
plt.ylabel('(Descendants-Ancestors)/(Descendants+Ancestors)')
#print(df['ancestors'])
plt.show()
"""
f,ax=plt.subplots(2,1,sharex=True)
ax[0].set_title('Ancestors')
ax[1].set_title('Descendants')
print(outa)
print(outb)
ax[0].hist(outa)
ax[1].hist(outb)
plt.show()
"""
"""
write_dot(G,'test.dot')
pos=graphviz_layout(G,prog='dot')
nx.draw(reg_ditree(5,2),pos)
plt.show()
"""

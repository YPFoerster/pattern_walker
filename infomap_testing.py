from infomap import Infomap
import random_walker as rw
import utils
import networkx as nx
import random
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt

offspring_factor=3
height=3
H=nx.generators.classic.balanced_tree(offspring_factor,height)
root=utils.list_degree_nodes(H,offspring_factor,1)[0]
H,_=utils.directify(H,root)
G=rw.patternWalker(H,root,15,0.01)

#add 10 additional random edges
for i in range(10):
    link=[random.choice( list(G.nodes) ),random.choice( list(G.nodes) )]
    G.add_edge(*link)



G.set_weights()

nx.write_pajek(G,"./outputs/20200916_infomap_test/pajekt_test.net")

pos=graphviz_layout(H,prog='dot')
nx.draw(G,pos)
nx.draw_networkx_labels(G,pos)
nx.draw_networkx_nodes(G,pos,nodelist=[root],node_color='r',labels=True)
plt.show()


#im=Infomap("--directed")
#im.read_file("./outputs/20200916_infomap_test/pajekt_test.net")
#im.run()

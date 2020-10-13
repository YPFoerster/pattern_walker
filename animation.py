import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from networkx.drawing.nx_agraph import graphviz_layout
import networkx as nx
import utils
import random_walker as rw

fig, ax = plt.subplots()
fig.set_tight_layout(True)

# Query the figure's on-screen size and DPI. Note that when saving the figure to
# a file, we need to provide a DPI for that separately.
print('fig size: {0} DPI, size in inches {1}'.format(
    fig.get_dpi(), fig.get_size_inches()))

def make_tree(lam,N,gamma,overlap):
    G,root=utils.poisson_ditree(lam)
    leaves = utils.leaves(G)
    r=len(list(G.successors(root)))
    print(r)
    sections=[i*int(N/r) for i in range(r+1)]
    sections=[(sections[i],min(sections[i+1]+overlap,N)) for i in range(len(sections[:-1]))]
    walker=rw.sectionedPatternWalker(G.copy(),root,N,gamma,sections)
    walker.set_weights()
    return G,root,walker

make_tree=utils.seed_decorator(make_tree,0)


#Create patternWalker based on the above tree.
pattern_len=30 #String/pattern length
flip_rate=5/pattern_len #Bit flipping rate for propagating patterns

offspring_factor=2.
overlap=10
H,root,G=make_tree(offspring_factor,pattern_len,flip_rate,overlap)

#G=rw.patternWalker(H,root,pattern_len,flip_rate,metric=metric)
#G.set_weights()
pos=graphviz_layout(G,prog='dot') #Positions can be taken from H
(edges,weights) = zip(*nx.get_edge_attributes(G,'weight').items())
nx.draw(G, pos, edgelist=edges, edge_color=weights, edge_cmap=plt.cm.Blues,ax=ax)
nx.draw_networkx_nodes(G,pos,nodelist=[root],node_color='r',ax=ax)
#Mark target node in green.
nx.draw_networkx_nodes(G,pos,nodelist=[G.target_node],node_color='g',ax=ax)


def update(i):
    ax.clear()
    label = 'timestep {0}'.format(i)
    print(label)

    nx.draw(G, pos, edgelist=edges, edge_color=weights, edge_cmap=plt.cm.Blues,ax=ax)
    nx.draw_networkx_nodes(G,pos,nodelist=[root],node_color='r',ax=ax)
    #Mark target node in green.
    nx.draw_networkx_nodes(G,pos,nodelist=[G.target_node],node_color='g',ax=ax)
    # Update the line and the axes (with a new xlabel). Return a tuple of
    # "artists" that have to be redrawn for this frame.
    nx.draw_networkx_nodes(G,pos,nodelist=[G.x],node_color='y', ax=ax)
    G.step()
    return ax

if __name__ == '__main__':
    anim = FuncAnimation(fig, update, frames=np.arange(0, 1000), interval=1)
    plt.show()
    """
    # FuncAnimation will call the 'update' function for each frame; here
    # animating over 10 frames, with an interval of 200ms between frames.
    anim = FuncAnimation(fig, update, frames=np.arange(0, 10), interval=200)
    if len(sys.argv) > 1 and sys.argv[1] == 'save':
        anim.save('line.gif', dpi=80, writer='imagemagick')
    else:
        # plt.show() will just loop the animation forever.
        plt.show()
    """

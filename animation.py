import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from networkx.drawing.nx_agraph import graphviz_layout
import networkx as nx
import utils
import random_walker as rw



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

def make_tree(lam,N,gamma,overlap):
    G,root=utils.poisson_ditree(lam)
    leaves = utils.leaves(G)
    walker=rw.sectionedPatternWalker(G.copy(),root,N,gamma,overlap)
    walker.set_weights()
    return G,root,walker

parser = argparse.ArgumentParser(description="""
    For a fixed target, on a fixed graph with fixed weigthts, we sample the
    first passage time starting from the root.
    """)

parser.add_argument("--lam", default=0.5, type=float, dest='branching_factor',
    help="Poisson branching factor (default: %(default)s)"
    )
parser.add_argument("--seed", default=0, type=int, dest='seed',
    help="Seed for random graph generation %(default)s"
    )
parser.add_argument("--gamma", default=0., type=float,
    help="Bit flipping rate (default: %(default)s)"
    )
parser.add_argument("--string-len", default=10, type=int, dest='string_len',
    help="Len of bit strings assigned to each node (default: %(default)s)"
    )
parser.add_argument("--overlap", default=0, type=float, dest='overlap',
    help="Fractional overlap of adjacent string sections (every sections reached out to the right over\
          the regular boundary index*int(string_length/num_sections). (default: %(default)s)")

parser.add_argument("--job-id", default="unkown", dest="job_id",
    help="SLURM job ID (default: %(default)s)"
    )

parser.add_argument("--job-name", default="unknown", dest="job_name",
    help="Job name as per submission to sbatch (default: %(default)s)"
    )

parser.add_argument("--output-dir", default=".", dest="out_dir", help="Output files will appear here. (default: %(default)s)")


args=parser.parse_args()

lam=args.branching_factor #offspring number
seed=args.seed
gamma=args.gamma #mutation rate
#bits in a pattern; must be adapted to ensure uniqueness of patterns
N=args.string_len
overlap=args.overlap
job_id=args.job_id #as assigned by SLURM, for instance
job_name=args.job_name #as submitted to SBATCH
out_dir=args.out_dir #where to dump all that output.
args=vars(args)
args['job_dir']=os.getcwd() #store the location of the script for rerefence
print(args)

os.chdir(out_dir) #This way, we can simply write files without specified paths.


fig, ax = plt.subplots(figsize=(16,9))
fig.set_tight_layout(True)

# Query the figure's on-screen size and DPI. Note that when saving the figure to
# a file, we need to provide a DPI for that separately.
print('fig size: {0} DPI, size in inches {1}'.format(
fig.get_dpi(), fig.get_size_inches()))



make_tree=utils.seed_decorator(make_tree,seed)


#Create patternWalker based on the above tree.
H,root,G=make_tree(lam,N,gamma,overlap)

#G=rw.patternWalker(H,root,pattern_len,flip_rate,metric=metric)
#G.set_weights()
pos=graphviz_layout(G,prog='dot') #Positions can be taken from H
(edges,weights) = zip(*nx.get_edge_attributes(G,'weight').items())
nx.draw(G, pos, edgelist=edges, edge_color=weights, edge_cmap=plt.cm.Blues,ax=ax)
nx.draw_networkx_nodes(G,pos,nodelist=[root],node_color='r',ax=ax)
#Mark target node in green.
nx.draw_networkx_nodes(G,pos,nodelist=[G.target_node],node_color='g',ax=ax)


anim = FuncAnimation(fig, update, frames=np.arange(0, 1000), interval=500)
anim.save('{job_name}_gamma_{gamma}_overlap_{overlap}.gif'.format(**args), dpi=80, writer='imagemagick')

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

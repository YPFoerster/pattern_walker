import random_walker as rw
from dag_exps import poisson_ditree
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

G,root=poisson_ditree(2.,20)
walker=rw.patternWalker(G,root,20,0.1)
walker.set_weights()
for i in range(1000):
    walker.step()
    print(walker.metric(walker.G.nodes[walker.x]['pattern'],walker.searched_pattern))

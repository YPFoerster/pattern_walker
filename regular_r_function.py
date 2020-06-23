"""
Just to plot a particular function related to a hierarchy measure on a regular tree.
"""

import matplotlib.pyplot as plt
import numpy as np

branching_number=3
num_gens=20

def h(l,b,n):
    if b==1:
        return n-2*l+1
    else:
        return b*(1-b**(n-l))/(1-b)-l+1

def w(l,b,n):
    if b==1:
        return n-1
    else:
        return b*(1-b**(n-l))/(1-b)+l-1

def r(l,b,n):
    return h(l,b,n)/w(l,b,n)

a=[ r(l,branching_number,num_gens) for l in range(1,num_gens+1) ]
plt.plot(a)
plt.show()

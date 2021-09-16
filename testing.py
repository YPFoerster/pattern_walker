import unittest
import pattern_walker as pw
from pattern_walker.utils import balanced_ditree,filter_nodes
from itertools import product
import multiprocessing as mp
import numpy as np
import pandas as pd

c=3 #offspring number
h=1
#bits in a pattern; must be adapted to ensure uniqueness of patterns
L=48
a=0.7
Gamma=0.4
beta_l=0.1
beta_h=0.0
increment=0.05

H,root=balanced_ditree(c,h)
#number of realisations of pattern distributions in this case
number_of_samples=2000
num_cores=4

Delta_range=range(0,int(L*(c-1)/(2*c))+1,5)
Gammap_range=np.arange(0,1+increment,increment)
param_range=product(Delta_range,Gammap_range)

class overlapTestCase(unittest.TestCase):


    def test_parts_overlap(self):
        mean_part_dist = pd.DataFrame(0.,index=Gammap_range,columns=Delta_range)
        control = pd.DataFrame(0.,index=Gammap_range,columns=Delta_range)


        if num_cores==1:
            for (Delta,Gammap) in param_range:
                out=avg_dist((Delta,Gammap))
                mean_part_dist[out[0]][out[1]]=out[2]
                control[out[0]][out[1]]=out[3]
        else:
            out_list=[]
            with mp.Pool(num_cores) as p:
                for out in p.map(avg_dist, param_range):
                    out_list.append(out)
            for out in out_list:
                mean_part_dist[out[0]][out[1]]=out[2]
                control[out[0]][out[1]]=out[3]

        testvalue=np.linalg.norm(mean_part_dist.values-control.values)/(len(Delta_range)*len(Gammap_range))
        self.assertTrue( testvalue<1e-3, testvalue )


def avg_dist(param_tuple):
    Delta=param_tuple[0]
    Gammap=param_tuple[1]

    a_h=(1-a)*Gammap+a-beta_h
    a_l=(1-a)*Gammap+beta_l
    G=pw.fullProbPatternWalker(H,root,L,a,a_l,a_h,Delta,Gamma,Gammap)
    part_dist=np.zeros(number_of_samples)
    parts=filter_nodes(G,'depth',1)
    parts=list(parts)

    for iter in range(number_of_samples):
        part_dist[iter]=np.mean([np.linalg.norm(G.nodes[parts[n1]]['pattern']-G.nodes[parts[n1+1]]['pattern'],ord=1) for n1 in range(len(parts)-1)] )
        G.reset_patterns()
    return [Delta,Gammap,np.mean(part_dist),theo_dist(Delta, Gammap)]



def theo_dist(Delta,Gammap):
    if beta_h==0:
        if Delta<=L*(c-2)/(2*c):
            return 2*L*Gammap*(1-Gammap)*(1-a)+2*L/c*(a-beta_l)*((c-2)*beta_l/a+1)-4*Delta*(a-beta_l)*beta_l/a
        else:
            return 2*L*Gammap*(1-Gammap)*(1-a)+2*L/c*(c-1)*(a-beta_l)-4*Delta*(a-beta_l)

    else:
        if Delta<=L*(c-2)/(2*c):
            return 2*L*Gammap*(1-Gammap)*(1-a)+2*L/c*(c*beta_l*(a-beta_l)/a+a+beta_l-beta_h-2*beta_l*(2*a-beta_l-beta_h)/a)+4*Delta/a*((a-beta_h)*beta_h-(a-beta_l)*beta_l)
        else:
            return 2*L*Gammap*(1-Gammap)*(1-a)-2*L/c*(a-beta_h)/a*( (beta_h+2*beta_l)*c - 2*(beta_h+beta_l) ) + 2*L/c*(c-1)*(a+beta_l-beta_h) + 4*Delta*(2/a*(a-beta_h)*(beta_h+beta_l)-a-beta_l+beta_h)


if __name__ == '__main__':
    unittest.main()

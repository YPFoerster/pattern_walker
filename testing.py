import unittest
import pattern_walker as pw
from pattern_walker.utils import balanced_ditree,leaves
import networkx as nx
from itertools import product
import multiprocessing as mp
import numpy as np
import pandas as pd

c=3 #offspring number
h=4
#bits in a pattern; must be adapted to ensure uniqueness of patterns
L=48
a=0.7
Gamma=0.4
Gammap=0.3
beta_l=0.1
beta_h=0.0
increment=0.05

H,root=balanced_ditree(c,h)
parts=nx.neighbors(H,root)
parts=list(parts)
leaves=leaves(H)
part_leaves=[[node for node in  nx.descendants(H,part) if node in leaves] for part in parts]
#number of realisations of pattern distributions in this case
number_of_samples=500
num_cores=4

class patternTestCase(unittest.TestCase):

    def test_parts_overlap(self):

        Delta_range=range(0,int(L*(c-1)/(2*c))+1,5)
        Gammap_range=np.arange(0,1+increment,increment)
        param_range=product(Delta_range,Gammap_range)

        mean_part_dist = pd.DataFrame(0.,index=Gammap_range,columns=Delta_range)
        control = pd.DataFrame(0.,index=Gammap_range,columns=Delta_range)


        if num_cores==1:
            for (Delta,Gammap) in param_range:
                out=avg_horizontal_dist((Delta,Gammap))
                mean_part_dist[out[0]][out[1]]=out[2]
                control[out[0]][out[1]]=out[3]
        else:
            out_list=[]
            with mp.Pool(num_cores) as p:
                for out in p.map(avg_horizontal_dist, param_range):
                    out_list.append(out)
            for out in out_list:
                mean_part_dist[out[0]][out[1]]=out[2]
                control[out[0]][out[1]]=out[3]

        testvalue=np.linalg.norm(mean_part_dist.values-control.values)/(len(Delta_range)*len(Gammap_range))
        self.assertTrue( testvalue<6e-3, testvalue )


    def test_tightness(self):

        Delta_range=range(0,int(L*(c-1)/(2*c))+1,5)
        Gamma_range=np.arange(0,1+increment,increment)
        param_range=product(Delta_range,Gamma_range)

        mean_vertical_dist = pd.DataFrame(0.,index=Gamma_range,columns=Delta_range)
        control = pd.DataFrame(0.,index=Gamma_range,columns=Delta_range)

        if num_cores==1:
            for (Delta,Gamma) in param_range:
                out=avg_vertical_dist((Delta,Gamma))
                mean_vertical_dist[out[0]][out[1]]=out[2]
                control[out[0]][out[1]]=out[3]
        else:
            out_list=[]
            with mp.Pool(num_cores) as p:
                for out in p.map(avg_vertical_dist, param_range):
                    out_list.append(out)
            for out in out_list:
                mean_vertical_dist[out[0]][out[1]]=out[2]
                control[out[0]][out[1]]=out[3]
        testvalue=np.linalg.norm(mean_vertical_dist.values-control.values)/(len(Delta_range)*len(Gamma_range))
        self.assertTrue( testvalue<6e-3, testvalue )




def avg_horizontal_dist(param_tuple):
    Delta=param_tuple[0]
    Gammap=param_tuple[1]

    a_h=(1-a)*Gammap+a-beta_h
    a_l=(1-a)*Gammap+beta_l
    G=pw.fullProbPatternWalker(H,root,L,a,a_l,a_h,Delta,Gamma,Gammap)
    part_dist=np.zeros(number_of_samples)

    for iter in range(number_of_samples):
        part_dist[iter]=np.mean([np.linalg.norm(G.nodes[parts[n1]]['pattern']-G.nodes[parts[n1+1]]['pattern'],ord=1) for n1 in range(len(parts)-1)] )
        G.reset_patterns()
    return [Delta,Gammap,np.mean(part_dist),theo_horizontal_dist(Delta, Gammap)]



def theo_horizontal_dist(Delta,Gammap):
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


def vertical_rate(Gamma,a_j,Delta):
    return 2*a_j*(1-a_j)*(1-(1-Gamma)**float(h-1))


def expected_vertical_distance(Gamma,Delta):
    a_l=(1-a)*Gammap+beta_l
    a_h=(1-a)*Gammap+a
    return vertical_rate(Gamma,a_h,Delta)*(L/c+2*Delta)+vertical_rate(Gamma,a_l,Delta)*(L-L/c-2*Delta)

def avg_vertical_dist(param_tuple):
    a_l=(1-a)*Gammap+beta_l
    a_h=(1-a)*Gammap+a
    Delta=param_tuple[0]
    Gamma=param_tuple[1]
    G=pw.fullProbPatternWalker(H,root,L,a,a_l,a_h,Delta,Gamma,Gammap)
    vertical_dist=0#np.zeros(number_of_samples)
    G.set_weights()
    #the following better in dicts of dicts (of dicts)
    for iter in range(number_of_samples):
        vertical_dist+=np.mean( [[ np.linalg.norm(G.nodes[parts[sec_ndx]]['pattern']-G.nodes[leaf]['pattern'],ord=1) for leaf in part_leaves[sec_ndx] ] for sec_ndx in range(c)] )
        G.reset_patterns()
    return (Delta,Gamma,vertical_dist/number_of_samples,expected_vertical_distance(Gamma,Delta))



if __name__ == '__main__':
    unittest.main()

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
increment=0.1
H,root=balanced_ditree(c,h)
number_of_samples=100
num_cores=4

class patternTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.Delta_range=range(0,int(L*(c-1)/(2*c))+1,5)
        cls.Gammap_range=np.arange(0,1+increment,increment)
        cls.Gamma_range=np.arange(0,1+increment,increment)
        cls.param_range=list(product(cls.Gammap_range,cls.Gamma_range,\
                cls.Delta_range))

        cls.testvalues=cls.get_testvalues(cls)

    def test_parts_overlap(self):
        test=np.array(self.testvalues[0])-np.array(self.testvalues[1])
        test=np.linalg.norm(test)/len(self.param_range)
        self.assertTrue(test<1e-3,test)

    def test_vertical_dist(self):
        test=np.array(self.testvalues[2])-np.array(self.testvalues[3])
        test=np.linalg.norm(test)/len(self.param_range)
        self.assertTrue(test<1e-3,test)


    def test_root_part_dist(self):
        test=np.array(self.testvalues[4])-np.array(self.testvalues[5])
        test=np.linalg.norm(test)/len(self.param_range)
        self.assertTrue(test<1e-3,test)

    def get_testvalues(self):
        mean_part_dist = []
        part_control = []
        mean_vertical_dist = []
        vertical_control = []
        mean_root_part_dist=[]
        root_part_control=[]
        if num_cores==1:
            for (Gammap,Gamma,Delta) in self.param_range:
                out=get_averages((Gammap,Gamma,Delta))
                mean_part_dist.append(out[0][0])
                part_control.append(out[0][1])
                mean_vertical_dist.append(out[1][0])
                vertical_control.append(out[1][1])
                mean_root_part_dist.append(out[2][0])
                root_part_control.append(out[2][1])
        else:
            out_list=[]
            with mp.Pool(num_cores) as p:
                for out in p.map(get_averages, self.param_range):
                    out_list.append(out)
            for out in out_list:
                mean_part_dist.append(out[0][0])
                part_control.append(out[0][1])
                mean_vertical_dist.append(out[1][0])
                vertical_control.append(out[1][1])
                mean_root_part_dist.append(out[2][0])
                root_part_control.append(out[2][1])

        return mean_part_dist,part_control,mean_vertical_dist,vertical_control,\
                mean_root_part_dist,root_part_control


def get_averages(args):
    (Gammap,Gamma,Delta)=args
    a_l=(1-a)*Gammap+beta_l
    a_h=(1-a)*Gammap+a-beta_h
    G=pw.patternStats(H,root,L,a,a_l,a_h,Delta,Gamma,Gammap)
    horizontal_sampled,vertical_sampled,root_part_sampled=\
        G.sample_distances(number_of_samples)
    horizontal_pred,vertical_pred,root_part_pred=\
        G.expected_part_dist(),G.expected_vertical_distance(),G.expected_root_to_part_distance()
    return [np.mean(horizontal_sampled),horizontal_pred],\
            [np.mean(vertical_sampled),vertical_pred],\
            [np.mean(root_part_sampled),root_part_pred]


if __name__ == '__main__':
    unittest.main()

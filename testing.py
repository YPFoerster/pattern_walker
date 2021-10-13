import unittest
import pattern_walker as pw
import mean_field as mf
from pattern_walker.utils import balanced_ditree,leaves,mfpt
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
Delta=0
beta_l=0.1
beta_h=0.0
increment=0.1
H,root=balanced_ditree(c,h)
leaves_list=leaves(H)
number_of_samples=100
eps=2e-2
num_cores=4

class WalkerDiffusionMFPTTestCase(unittest.TestCase):
    def test_mfpt(self):
        G=pw.walker(H,root,1.)
        G.set_weights()
        true_mfpt=mfpt(G,[(root,leaves_list[0])])
        diffusive_mfpt=h*(2*c**(h+1)/(c-1)-1)-2*c*(c**h-1)/(c-1)**2
        self.assertTrue(abs(true_mfpt-diffusive_mfpt)<eps,'true_mfpt: {},diffusive_mfpt: {}'.format(true_mfpt,diffusive_mfpt))

class patterWalkerDiffusionMFPTTestCase(unittest.TestCase):
    def test_mfpt(self):
        G=pw.patternWalker(H,root,L,0.)
        G.set_weights()
        true_mfpt=mfpt(G,[(G.root,G.target_node)])
        diffusive_mfpt=h*(2*c**(h+1)/(c-1)-1)-2*c*(c**h-1)/(c-1)**2
        self.assertTrue(abs(true_mfpt-diffusive_mfpt)<eps,'true_mfpt: {},diffusive_mfpt: {}'.format(true_mfpt,diffusive_mfpt))

class fullprobDiffusionMFTPTTestCase(unittest.TestCase):

    def test_mfpt(self):
        G=pw.patternStats(H,root,L,0.,0.,0.,Delta,Gamma,0.)
        G.set_weights()
        true_mfpt=mfpt(G,[(G.root,G.target_node)])
        diffusive_mfpt=h*(2*c**(h+1)/(c-1)-1)-2*c*(c**h-1)/(c-1)**2
        self.assertTrue(abs(true_mfpt-diffusive_mfpt)<eps,'true_mfpt: {},diffusive_mfpt: {}'.format(true_mfpt,diffusive_mfpt))


class MeanFieldDiffusionMFTPTTestCase(unittest.TestCase):
    def test_MF_mfpt(self):
        G=mf.MF_patternWalker(c,h,H,root,L,0.,0.,0.,Delta,Gamma,0.)
        G.set_weights()
        MF_mfpt=G.MF_mfpt()
        diffusive_mfpt=h*(2*c**(h+1)/(c-1)-1)-2*c*(c**h-1)/(c-1)**2
        self.assertTrue(abs(MF_mfpt-diffusive_mfpt)<eps,'MF_mfpt: {},diffusive_mfpt: {}'.format(MF_mfpt,diffusive_mfpt))

    def test_MTM_mfpt(self):
        G=mf.MF_patternWalker(c,h,H,root,L,0.,0.,0.,Delta,Gamma,0.)
        G.set_weights()
        MF_mfpt=G.MTM_mfpt(number_of_samples)
        diffusive_mfpt=h*(2*c**(h+1)/(c-1)-1)-2*c*(c**h-1)/(c-1)**2
        self.assertTrue(abs(MF_mfpt-diffusive_mfpt)<eps,'MF_mfpt: {},diffusive_mfpt: {}'.format(MF_mfpt,diffusive_mfpt))

    def test_MTM_approx_mfpt(self):
        G=mf.MF_patternWalker(c,h,H,root,L,0.,0.,0.,Delta,Gamma,0.)
        G.set_weights()
        MF_mfpt=G.MTM_mfpt(0)
        diffusive_mfpt=h*(2*c**(h+1)/(c-1)-1)-2*c*(c**h-1)/(c-1)**2
        self.assertTrue(abs(MF_mfpt-diffusive_mfpt)<eps,'MF_mfpt: {},diffusive_mfpt: {}'.format(MF_mfpt,diffusive_mfpt))

class OverlapMeanFieldDiffusionMFTPTTestCase(unittest.TestCase):
    def test_MF_mfpt(self):
        G=mf.overlap_MF_patternWalker(c,h,H,root,L,0.,0.,0.,Delta,Gamma,0.)
        G.set_weights()
        MF_mfpt=G.MF_mfpt()
        diffusive_mfpt=h*(2*c**(h+1)/(c-1)-1)-2*c*(c**h-1)/(c-1)**2
        self.assertTrue(abs(MF_mfpt-diffusive_mfpt)<eps,'MF_mfpt: {},diffusive_mfpt: {}'.format(MF_mfpt,diffusive_mfpt))

    def test_MTM_mfpt(self):
        G=mf.overlap_MF_patternWalker(c,h,H,root,L,0.,0.,0.,Delta,Gamma,0.)
        G.set_weights()
        MF_mfpt=G.MTM_mfpt(number_of_samples)
        diffusive_mfpt=h*(2*c**(h+1)/(c-1)-1)-2*c*(c**h-1)/(c-1)**2
        self.assertTrue(abs(MF_mfpt-diffusive_mfpt)<eps,'MF_mfpt: {},diffusive_mfpt: {}'.format(MF_mfpt,diffusive_mfpt))

    def test_MTM_approx_mfpt(self):
        G=mf.overlap_MF_patternWalker(c,h,H,root,L,0.,0.,0.,Delta,Gamma,0.)
        G.set_weights()
        MF_mfpt=G.MTM_mfpt(0)
        diffusive_mfpt=h*(2*c**(h+1)/(c-1)-1)-2*c*(c**h-1)/(c-1)**2
        self.assertTrue(abs(MF_mfpt-diffusive_mfpt)<eps,'MF_mfpt: {},diffusive_mfpt: {}'.format(MF_mfpt,diffusive_mfpt))


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
        self.assertTrue(test<eps,test)

    def test_vertical_dist(self):
        test=np.array(self.testvalues[2])-np.array(self.testvalues[3])
        test=np.linalg.norm(test)/len(self.param_range)
        self.assertTrue(test<eps,test)


    def test_root_part_dist(self):
        test=np.array(self.testvalues[4])-np.array(self.testvalues[5])
        test=np.linalg.norm(test)/len(self.param_range)
        self.assertTrue(test<eps,test)

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

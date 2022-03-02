import unittest
import pattern_walker as pw
import mean_field as mf
from pattern_walker.utils import balanced_ditree,leaves,mfpt,eq_prob_cluster_ratios,cluster_by_branch
import networkx as nx
from itertools import product
import multiprocessing as mp
import numpy as np
import pandas as pd

c=3 #offspring number
h=4
#bits in a pattern; must be adapted to ensure uniqueness of patterns
L=16*c
Delta_max=int(L*(c-1)/(2*c))
a=1.
Gamma=2.
Gammap=0.3
Delta=4
beta_l=a/10
beta_h=0.0
a_l=(1-a)*Gammap+beta_l
a_h=(1-a)*Gammap+a-beta_h

increment=0.1
H,root=balanced_ditree(c,h)
leaves_list=leaves(H)
number_of_samples=10
eps=5e-3
num_cores=2

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

class MeanFieldMFPTMethodsTestCase(unittest.TestCase):

    def test_MF_mfpt_max_overlap(self):
        G=mf.MF_patternWalker(c,h,H,root,L,a,a_l,a_h,Delta_max,Gamma,Gammap)
        G.set_weights()
        G_O=mf.overlap_MF_patternWalker(c,h,H,root,L,a,a_l,a_h,Delta_max,Gamma,Gammap)
        G_O.set_weights()
        MF_mfpt=G.MF_mfpt()
        overlap_MF_mfpt=G_O.MF_mfpt()
        self.assertTrue(abs(MF_mfpt-overlap_MF_mfpt)<eps,\
            'at full overlap, we shoud have MF_mfpt ignoring overlap: {} = MF_mfpt respecting overlap: {}'.format(MF_mfpt,overlap_MF_mfpt))

    def test_MF_mfpt_wo_overlap(self):
        G=mf.MF_patternWalker(c,h,H,root,L,a,a_l,a_h,Delta,Gamma,Gammap)
        G.set_weights()
        MF_mfpt=G.MF_mfpt()
        MTM_approx_mfpt=G.MTM_mfpt(0)
        self.assertTrue(abs(MF_mfpt-MTM_approx_mfpt)<eps,\
            'ignoring overlap: MF_mfpt: {},MF_mfpt from MTM: {}'.format(MF_mfpt,MTM_approx_mfpt))

    def test_MF_mfpt(self):
        G=mf.overlap_MF_patternWalker(c,h,H,root,L,a,a_l,a_h,Delta,Gamma,Gammap)
        G.set_weights()
        MF_mfpt=G.MF_mfpt()
        MTM_approx_mfpt=G.MTM_mfpt(0)
        self.assertTrue(abs(MF_mfpt-MTM_approx_mfpt)<eps,\
            'respecting overlap: MF_mfpt: {},MTM_mfpt from MTM: {}'.format(MF_mfpt,MTM_approx_mfpt))

    def test_MTM_graph_nodes(self):
        M=mf.MF_patternWalker(c,h,H,root,L,a,a_l,a_h,Delta,Gamma,Gammap)
        M.set_weights()
        M_G,_=M.approx_MTM()
        test_list=[(node in M_G.nodes) for node in M.nodes]
        self.assertTrue( all(test_list) )

    def test_MTM_graph_nodes_reverse(self):
        M=mf.MF_patternWalker(c,h,H,root,L,a,a_l,a_h,Delta,Gamma,Gammap)
        M.set_weights()
        M_G,_=M.approx_MTM()
        test_list=[(node in M.nodes) for node in M_G.nodes]
        self.assertTrue( all(test_list) )


class MeanFieldEqRatiosTestCase(unittest.TestCase):

    def test_MF_local_eq_wo_overlap(self):
        M=mf.MF_patternWalker(c,h,H,root,L,a,a_l,a_h,Delta,Gamma,Gammap)
        M.set_weights()
        M_G,_=M.approx_MTM()
        clusters=cluster_by_branch(M)
        Q=eq_prob_cluster_ratios(M_G,clusters,list(clusters.keys()))
        Q_test=np.array([M.root_cluster_eq_ratio(),M.sub_root_cluster_eq_ratio()]+[M.eq_ratio(h-k) for k in range(2,h+1)])
        self.assertTrue(np.linalg.norm(Q-Q_test)<eps,np.linalg.norm(Q-Q_test))


    def test_MF_local_eq_with_overlap(self):
        M=mf.overlap_MF_patternWalker(c,h,H,root,L,a,a_l,a_h,Delta,Gamma,Gammap)
        M.set_weights()
        M_G,_=M.approx_MTM()
        clusters=cluster_by_branch(M)
        Q=eq_prob_cluster_ratios(M_G,clusters,list(clusters.keys()))
        Q_test=np.array([M.root_cluster_eq_ratio(),M.sub_root_cluster_eq_ratio()]+[M.eq_ratio(h-k) for k in range(2,h+1)])
        self.assertTrue(np.linalg.norm(Q-Q_test)<eps,np.linalg.norm(Q-Q_test))



class MeanFieldOverlapNumbersTestCase(unittest.TestCase):
    def test_max_Delta_high_marginal(self):
        G=mf.overlap_MF_patternWalker(c,h,H,root,L,a,a_l,a_h,Delta_max,Gamma,Gammap)
        self.assertEqual(G.O_hh,1.)

    def test_max_Delta_low_marginal(self):
        G=mf.overlap_MF_patternWalker(c,h,H,root,L,a,a_l,a_h,Delta_max,Gamma,Gammap)
        self.assertEqual(G.O_ll,0.)

    def test_min_Delta_high_marginal(self):
        G=mf.overlap_MF_patternWalker(c,h,H,root,L,a,a_l,a_h,0,Gamma,Gammap)
        self.assertEqual(G.O_hh,0.)

    def test_min_Delta_low_marginal(self):
        G=mf.overlap_MF_patternWalker(c,h,H,root,L,a,a_l,a_h,0,Gamma,Gammap)
        self.assertTrue(abs(G.O_ll-1+2/c)<eps,abs(G.O_ll-1+2/c))

class patternTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.Delta_range=range(0,int(L*(c-1)/(2*c))+1,int(L*(c-1)/(8*c)))
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

test_case_dict={
    'diffusion_test_cases':(WalkerDiffusionMFPTTestCase,\
    patterWalkerDiffusionMFPTTestCase,fullprobDiffusionMFTPTTestCase,\
    MeanFieldDiffusionMFTPTTestCase,OverlapMeanFieldDiffusionMFTPTTestCase),
    'mean_field_test_cases':(MeanFieldMFPTMethodsTestCase,MeanFieldOverlapNumbersTestCase,MeanFieldEqRatiosTestCase),
    #'patternStats_test_cases':(patternTestCase,)
    }

def load_tests(loader, test_cases):
    suite=unittest.TestSuite()
    for test_class in test_cases:
        tests=loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    return suite

if __name__ == '__main__':
    #unittest.main()

    for test_cases in test_case_dict.values():
        loader=unittest.TestLoader()
        test_suite=load_tests(loader,test_cases)
        runner=unittest.TextTestRunner(verbosity=3)
        runner.run(test_suite)

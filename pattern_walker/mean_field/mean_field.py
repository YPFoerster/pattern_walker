"""
Class for walker with approximate average-patterns, in particular the
approximate MFPT from root to target.
"""

import pattern_walker as rw
import numpy as np
import networkx as nx

__all__ = [
    'MF_patternWalker', 'overlap_MF_patternWalker','MFPropertiesCAryTrees',\
    'MF_mfpt_cary_tree','MF_patternWalker_general'
    ]


class MF_patternWalker(rw.fullProbPatternWalker):
    """
    Have mean-field computation for pattern_walker all in one place. args and
    kwargs are the relevant parameters for the patterwalker.
    """
    def __init__(self,c=4,h=3,*args,**params):
        self.c=c
        self.h=h
        super(MF_patternWalker,self).__init__(*args,**params)
        self.set_weights()
        self.set_coordinates()

    def Q_power(self,k,aj=None):
        """
         in-part transitition matrix for pattern bits, over edge-distance k
        """
        if aj is None:
            aj=self.a_high
        Gamma=self.Gamma

        if aj>0:
            out= np.array(
                    [[1-aj*(1-(1-Gamma)**k),aj*(1-(1-Gamma)**k)],
                             [(1-aj)*(1-(1-Gamma)**k),1-(1-aj)*(1-(1-Gamma)**k)]]
                    )
        elif aj==0:
            out=np.linalg.matrix_power(np.array( [[1,0],[Gamma,1-Gamma]] ),k)
        return out

    def R_up(self,aj=None,a=None,Gammap=None):
        """
        part-to-root transition matrix for pattern bit
        """
        if aj is None:
            ajl=self.a_high
        if a is None:
            a=self.a_root
        if Gammap is None:
            Gammap=self.Gamma_root

        if aj>0 and aj<1:
            out= np.array(
                    [[(1-a)*(1-Gammap)/(1-aj),1-(1-a)*(1-Gammap)/(1-aj)],
                     [(1-a)/aj*Gammap,1-(1-a)/aj*Gammap]]
                    )
        elif aj==0:
            out=np.array(
                    [[(1-a)*(1-Gammap),1-(1-a)*(1-Gammap)],
                     [0.,1.]]
                    )
        elif aj==1:
            out=np.array(
                    [[0.,1.],
                     [0.,1.]]
                    )
        return out

    def R_down(self,aj=None,a=None,Gammap=None):
        """
        root-to-part transition matrix for pattern bit
        """
        if aj is None:
            aj=self.a_high
        if a is None:
            a=self.a_root
        if Gammap is None:
            Gammap=self.Gamma_root

        if a>0:
            return np.array(
                    [[1-Gammap,Gammap],
                     [1-(aj-(1-a)*Gammap)/a,(aj-(1-a)*Gammap)/a]]
                    )
        elif a==0:
            out=np.array(
                    [[1-Gammap,Gammap],
                     [0.,1.]]
                    )
        return out


    def f(self,k,up=0,down=0,m=0,mu=0,ajl=None,ajr=None):
        """
        for full overlap, the pattern distance rates at graph distance k from the target. Cases separated depending
        whether the root needs to be involved
        k includes the upward root link if applicable, m includes the downwards root link, if applicable
        so the sum of k and m has to be the graph distance
        """
        if ajl is None:
            ajl=self.a_high
        if ajr is None:
            ajr=self.a_high
        a=self.a_root
        Gamma=self.Gamma
        Gammap=self.Gamma_root
        out=0.

        if k>0 and up+down+m==0:
            out=2*ajl*(1-ajl)*(1-(1-Gamma)**k)

        elif up==1 and down+m==0:
            if a:
                out=a+ajl-2*a*ajl+2*(1-a)*(1-Gamma)**k*(Gammap-ajl)
            else:
                out=Gammap

        elif up==down==1:
            if a:
                out=2/a*(1-Gamma)**(k+m)*((1-a)*Gammap*(ajl+ajr-Gammap)-ajl*ajr)+ajl+ajr-2*ajl*ajr*(1-(1-Gamma)**(k+m))
            else:
                out=2*(1-Gammap)*(Gammap-a*(1-Gamma)**m)+a*(1-Gammap)*(1-Gamma)**m



        elif up==0 and down==1:
            if a:
                out=2*a*(1-a)*(1-(1-Gamma)**m)+2*(1-a)*(1-Gamma)**m*Gammap
            else:
                out=Gammap

        elif down==0:
            out=2*ajr*(1-ajr)*(1-(1-Gamma)**m)

        return out

    def epsilon(self,f2,fk,fk2):
        L=self.pattern_len
        out=0.
        if fk==0:
            if f2==0:
                out=0.
            else:
                out=f2*self.pattern_len
        else:
            dk1_inv_exp=(1-(1-fk)**(self.pattern_len+1))/((self.pattern_len+1)*fk)
            out=-1+(1+L*f2*fk2/(1-fk)-fk2*(1-fk-f2)/(fk*(1-fk)))*dk1_inv_exp+fk2*(1-fk-f2)/(fk*(1-fk))

        return out

    def root_cluster_eq_ratio(self):
        #eq prob of cluster divided by eq prob of articulation pt, here the root itself
        bias_list=[
                1+self.epsilon(
                    self.f(0,1,1,0),\
                    self.f(self.h-1,0,0,0),\
                    self.f(self.h-1,1,1,0)
                    ),\
                1+self.epsilon(
                    self.f(0,0,1,1),\
                    self.f(self.h-1,1,0,0),\
                    self.f(self.h-1,1,1,1))
            ]+[
                1+self.epsilon(
                    self.f(0,0,0,2),\
                    self.f(self.h-1,1,1,m),\
                    self.f(self.h-1,1,1,m+2) )
            for m in range(self.h-1)
            ]
        out=1+(self.c-1)/(self.c-1+bias_list[0])*\
            (
            np.sum([
                    self.c**l*(self.c+bias_list[l+1])/\
                    np.prod( [bias_list[k+1] for k in range(l+1)])
                for l in range(self.h-1)
                ])+\
            self.c**(self.h-1)/np.prod( [bias_list[l+1] for l in range(self.h-1)])
            )
        return out

    def sub_root_cluster_eq_ratio(self):
        #eq prob of cluster divided by eq prob of articulation pt, here the node under the root

        #just under the root we need to distinguish two different paths, namely,
        # the one going over the root and the ones staying in the target-part
        e_r=1+self.epsilon(self.f(2,0,0,0),self.f(self.h-2,0,0,0),self.f(self.h,0,0,0))
        e_u=1+self.epsilon(self.f(1,1,0,0),self.f(self.h-2,0,0,0),self.f(self.h-1,1,0,0))

        bias_list=[ e_r,e_u ]+[ 1+self.epsilon( self.f(2,0,0,0),self.f(self.h-1+l,0,0,0),self.f(self.h+1+l,0,0,0) )  for l in range(self.h-2) ]
        out=1+(self.c-1)*e_u/( (self.c-1)*e_u+e_r+e_r*e_u )*\
            (
            np.sum([
                    self.c**l*(self.c+bias_list[l+2])/np.prod( [bias_list[m+2] for m in range(l+1)])
                for l in range(self.h-2)
                ])+\
            self.c**(self.h-2)/np.prod( [bias_list[l+2] for l in range(self.h-2)])
            )
        return out

    def eq_ratio(self,k):
        #eq prob of cluster divided by eq prob of articulation pt at height k over target

        if k==0:
            return 1.
        else:
            bias_list=[ 1+self.epsilon( self.f(2,0,0,0),self.f(k-1+l,0,0,0),self.f(k+1+l,0,0,0) )  for l in range(k) ]
            out=1+ (self.c-1)/(self.c+bias_list[0])*\
                (
                np.sum([
                        self.c**l*(self.c+bias_list[l+1])/np.prod( [ bias_list[m+1] for m in range(l+1)])
                    for l in range(k-1)
                    ])+\
                self.c**(k-1)/np.prod( [ bias_list[l+1] for l in range(k-1)])
                )
        return out

    def MF_mfpt(self,ajl=None,ajr=None,a=None,Gamma=None,Gammap=None,**kwargs):
        """
        calculate C_MF based on the functions defined above.
        # TODO: remove arguments other than self as none of them are needed
        """
        #just under the root we need to distinguish two different paths, namely,
        # the one going over the root and the ones staying in the target-part
        e_r=1+self.epsilon(self.f(2,0,0,0),self.f(self.h-2,0,0,0),self.f(self.h,0,0,0))
        e_u=1+self.epsilon(self.f(1,1,0,0),self.f(self.h-2,0,0,0),self.f(self.h-1,1,0,0))
        cord_weight_list =\
            [
            1+self.epsilon(
                    self.f(0,1,1,0),\
                    self.f(self.h-1,0,0,0),\
                    self.f(self.h-1,1,1,0)
                )
            ]+\
            [ e_u ]+\
            [
                 1+self.epsilon(
                    self.f(2,0,0,0),\
                    self.f(self.h-k-1,0,0,0),\
                    self.f(self.h-k+1,0,0,0)
                    )
            for k in range(2,self.h+1)
            ]
        numerators=[ self.c-1+cord_weight_list[0] ]+[(e_u*(self.c-1)+e_r+e_r*e_u)/e_r]+[self.c+cord_weight_list[k] for k in range(2,self.h+1)]
        eq_ratio_list = [self.root_cluster_eq_ratio(), self.sub_root_cluster_eq_ratio()]+[ self.eq_ratio(self.h-k) for k in range(2,self.h+1) ]
        out = np.sum( [np.sum([eq_ratio_list[k]*numerators[k]/np.prod(cord_weight_list[k:i]) for k in range(i)]) for i in range(self.h+1)] )
        return out


    def MTM(self, number_samples: int,nodelist: list=None) ->np.array:
        #return the average transition matrix, sampled over number_samples interations.
        W=np.zeros( (len(self),len(self)) )
        if nodelist is None:
            nodelist=[self.root]+list( self.nodes -set([self.root,self.target_node])  )+[self.target_node]
        for _ in range(number_samples):
            self.reset_patterns()
            W_temp=nx.to_numpy_array(self,nodelist=nodelist)
            if (np.sum(W_temp,axis=-1)!=1).any:
                W_temp=np.diag(1/np.sum(W_temp,axis=-1)).dot(W_temp)
            W+=W_temp
        W/=number_samples
        return W

    def MTM_mfpt(self, number_samples: int=0, nodelist: list=None) -> np.float():
        """
        Calculates MFPT based on mean transition matrix, MTM. If number_samples=0,
        the approximate function approx_MTM is used. Else we sample the MTM.
        """
        out=0
        if nodelist is None:
            nodelist=[self.root]+list( self.nodes -set([self.root,self.target_node])  )+[self.target_node]

        if number_samples:
            out=self.MTM(number_samples,nodelist=nodelist)

        else:
            _,out=self.approx_MTM(nodelist=nodelist)
        out = np.sum( np.linalg.inv( np.eye(len(self)-1) - out[:-1,:-1] ),axis=-1 )[0]
        return out

    def approx_MTM(self,nodelist=None):
        """
        return approximate mean transition matrix calculated analytically
        """
        if nodelist is None:
            nodelist=[self.root]+list( self.nodes -set([self.root,self.target_node])  )+[self.target_node]
        out=nx.DiGraph()
        out.add_nodes_from(self.nodes)
        for node in self.nodes:
            weights=self.mean_out_weights(node)
            out.add_weighted_edges_from(weights)
        return out,nx.to_numpy_array(out,nodelist=nodelist)

    def mean_out_weights(self,node):
        """
        return (approximate) mean weights of edges leaving node.
        determines path 5-label and distinguishes cases accordingly
        """
        neigh=list(self.neighbors(node))
        out=[]
        try:
            toward_target=nx.shortest_path(self,node,self.target_node)[1]
        except IndexError:
            pass

        if len(neigh)== 1:
            out=[ (node,*neigh,1.) ]

        elif node==self.root:
            #the mu's aren't strictly required here, but we need them for
            #the overlap case
            weights=[(1+\
                self.epsilon(
                    self.f(0,1,1,0,mu),\
                    self.f(self.h-1,0,0,0,mu),\
                    self.f(self.h-1,1,1,0,mu))
                )
                for mu in range(2,self.c+1)
                ]
            e_0=np.prod(weights)
            normalisation=1/(e_0+\
                np.sum([ e_0/weight for weight in weights ])
                )
            #counter-clockwise. First towards target, then the other parts in order
            out.append((node,toward_target,e_0*normalisation))
            for neighbor in set(neigh)-set([toward_target]):
                part=self.nodes[neighbor]['coordinates'][4]
                out.append((node,neighbor,e_0*normalisation/weights[part-2]))


        elif self.root in neigh and self.nodes[node]['coordinates'][2]==0:
            e_r=1+self.epsilon(self.f(2,0,0,0),self.f(self.h-2,0,0,0),self.f(self.h,0,0,0))
            e_u=1+self.epsilon(self.f(1,1,0,0),self.f(self.h-2,0,0,0),self.f(self.h-1,1,0,0))
            normalisation=1/(e_u*(self.c-1)+e_r+e_r*e_u)
            for neighbor in neigh:
                #counter-clockwise, starting from the target, then other children,
                #finally root
                if neighbor ==toward_target:
                    out.append( (node,neighbor,e_u*e_r*normalisation ) )
                elif not neighbor==self.root:
                    out.append( (node,neighbor,e_u*normalisation ) )
                else:
                    out.append( (node,neighbor,e_r*normalisation ) )

        elif self.root in neigh:
            coordinates=list(self.nodes[node]['coordinates'])
            coordinates[2]=0
            e=1+self.epsilon(self.f(0,0,1,1,coordinates[4]),self.f(self.h-1,1,0,0,coordinates[4]),self.f(self.h-1,1,1,1,coordinates[4]))
            for neighbor in neigh:
                #counter-clockwise. first root (=targetwards), then children
                if neighbor==self.root:
                        out.append( (node,neighbor, e/(self.c+e) ) )
                else:
                    out.append( ( node,neighbor,1/(self.c+e) ) )

        else:
            coordinates=list(self.nodes[node]['coordinates'])
            short_path=[2,0,0,0]
            if coordinates[3]>0:
                coordinates[3]-=1
                short_path=[0,0,0,2]
            else:
                coordinates[0]-=1
            e=1+self.epsilon(self.f(*short_path,coordinates[-1]),self.f( *coordinates ),self.f( *[coordinates[i]+short_path[i] for i in range(4)] ))
            # TODO: following line with tightness 0/1 suitable for unit test
            #print('e:',e,self.nodes[node]['coordinates'],self.f(*coordinates))
            for neighbor in neigh:
                #counter-clockwise. first targetwards, then children
                if neighbor==toward_target:
                    out.append(( node,neighbor,e/(self.c+e)))
                else:
                    out.append(( node,neighbor,1/(self.c+e) ))
        return out

    def diffusive_mfpt(self):
        #Return MFPT for diffusive walker on the same graph
        h=self.h
        c=self.c
        return h*(2*c**(h+1)/(c-1)-1)-2*c*(c**h-1)/(c-1)**2


class overlap_MF_patternWalker(MF_patternWalker):
    """
    class to incorporate overlap into MF_patternwalker methods. defines a new f
    using MF_patternwalker.f, overwrites root_cluster_eq_ratio (as different parts
    have different overlaps with target-part) and MF_mfpt (same reason)

    """
    def __init__(self,c=4,h=3,*args,**params):
        self.c=c
        self.h=h
        super(overlap_MF_patternWalker,self).__init__(c,h,*args,**params)

        if self.overlap>self.pattern_len*(self.c-1)/(2*self.c):
            self.overlap=(self.pattern_len-int(self.pattern_len/self.c))/2
        self.O_list=np.array([
            max(0,int(self.pattern_len/self.c)*(2-i)+2*self.overlap)+\
            max(0,i*int(self.pattern_len/self.c)-self.pattern_len+2*self.overlap)
            for i in range(2,c+1)])
        self.U_list=np.array([
            max(0,-int(self.pattern_len/self.c)*(2-i)-2*self.overlap)+\
            max(0,-i*int(self.pattern_len/self.c)+self.pattern_len-2*self.overlap)
            for i in range(2,c+1)])
        self.O_hh=np.sum(self.O_list)/(self.pattern_len*(self.c-1))
        self.O_ll=np.sum(self.U_list)/(self.pattern_len*(self.c-1))
        self.O_hl=(1-self.O_hh-self.O_ll)/2
        self.O_lh=self.O_hl

    def f(self,k,up,down,m,mu=2,**kwargs):
        a_h=self.a_high
        a_l=self.a_low
        a=self.a_root
        Gamma=self.Gamma
        Gammap=self.Gamma_root

        out=0.
        coordinates=(k,up,down,m)

        if up==0 and down+m==0:
            #target branch
            f_h=MF_patternWalker.f(self,*coordinates,ajl=a_h,ajr=a_h)
            f_l=MF_patternWalker.f(self,*coordinates,ajl=a_l,ajr=a_l)
            out=f_h*(self.pattern_len/self.c+2*self.overlap)/self.pattern_len+\
            f_l*(
                self.pattern_len-self.pattern_len/self.c-2*self.overlap
                )/self.pattern_len

        elif up==1 and down==0:
            #target branch up to root
            f_h=MF_patternWalker.f(self,*coordinates,ajl=a_h,ajr=a)
            f_l=MF_patternWalker.f(self,*coordinates,ajl=a_l,ajr=a)
            out=f_h*(self.pattern_len/self.c+2*self.overlap)/self.pattern_len+\
                f_l*(
                    self.pattern_len-self.pattern_len/self.c-2*self.overlap
                    )/self.pattern_len

        elif up==0 and down==1:
            #root down to non-target branch
            f_h=MF_patternWalker.f(self,*coordinates,ajl=a,ajr=a_h)
            f_l=MF_patternWalker.f(self,*coordinates,ajl=a,ajr=a_l)
            out=f_h*(self.pattern_len/self.c+2*self.overlap)/self.pattern_len+\
                f_l*(
                    self.pattern_len-self.pattern_len/self.c-2*self.overlap
                    )/self.pattern_len

        elif up+k==0 and down==0:
            #non-targget branch
            f_h=MF_patternWalker.f(self,*coordinates,ajl=a_h,ajr=a_h)
            f_l=MF_patternWalker.f(self,*coordinates,ajl=a_l,ajr=a_l)
            out=f_h*(self.pattern_len/self.c+2*self.overlap)/self.pattern_len+\
                f_l*(
                    self.pattern_len-self.pattern_len/self.c-2*self.overlap
                    )/self.pattern_len

        else:
            #from target branch over root to non-target branch
            f_hh=MF_patternWalker.f(self,*coordinates,ajl=a_h,ajr=a_h)
            f_hl=MF_patternWalker.f(self,*coordinates,ajl=a_h,ajr=a_l)
            f_lh=MF_patternWalker.f(self,*coordinates,ajl=a_l,ajr=a_h)
            f_ll=MF_patternWalker.f(self,*coordinates,ajl=a_l,ajr=a_l)
            if mu==0:
                #is this the case where we don't care?
                out=self.O_hh*f_hh+self.O_lh*f_lh+self.O_hl*f_hl+self.O_ll*f_ll
            else:
                out=self.O_list[mu-2]/self.pattern_len*f_hh+\
                    (self.pattern_len-self.U_list[mu-2]-self.O_list[mu-2])/self.pattern_len*\
                    (f_hl+f_lh)/2+self.U_list[mu-2]/self.pattern_len*f_ll

        return out

    def root_cluster_eq_ratio(self):
        #eq prob of cluster divided by eq prob of articulation pt, here the root itself

        branch_weights=[(1+\
            self.epsilon(
                self.f(0,1,1,0,mu),\
                self.f(self.h-1,0,0,0,mu),\
                self.f(self.h-1,1,1,0,mu))
            )
            for mu in range(2,self.c+1)
            ]

        e_0=np.prod(branch_weights)
        e_r_list=[e_0/weight for weight in branch_weights]
        branch_normalisation=1/(e_0+np.sum(e_r_list))
        bias_dict={
            mu: [e_0,e_r_list[mu-2], 1+self.epsilon(self.f(0,0,1,1,mu),\
                self.f(self.h-1,1,0,0,mu),self.f(self.h-1,1,1,1,mu))]\
                +[
                    1+self.epsilon(
                        self.f(0,0,0,2,mu),\
                        self.f(self.h-1,1,1,l,mu),\
                        self.f(self.h-1,1,1,l+2,mu)
                    )
                for l in range(self.h-2)
                ]
            for mu in range(2,self.c+1)
            }

        out=1+np.sum([branch_normalisation*bias_dict[mu][1]*\
                (
                np.sum([
                    self.c**l*(self.c+bias_dict[mu][l+2])/\
                    np.prod([bias_dict[mu][k+2] for k in range(l+1)])
                for l in range(self.h-1)
                ])+\
                self.c**(self.h-1)/\
                    np.prod([bias_dict[mu][l+2] for l in range(self.h-1)])
                )
            for mu in range(2,self.c+1)
            ])

        return out

    def MF_mfpt(self,ajl=None,ajr=None,a=None,Gamma=None,Gammap=None,**kwargs):
        #just under the root we need to distinguish two different paths, namely,
        # the one going over the root and the ones staying in the target-part

        branch_weights=[(1+\
            self.epsilon(
                self.f(0,1,1,0,mu),\
                self.f(self.h-1,0,0,0,mu),\
                self.f(self.h-1,1,1,0,mu))
            )
            for mu in range(2,self.c+1)
            ]
        e_root=np.prod(branch_weights)

        e_root_norm=e_root+np.sum([
                e_root/weight for weight in branch_weights])

        e_r=1+self.epsilon(
            self.f(2,0,0,0),\
            self.f(self.h-2,0,0,0),\
            self.f(self.h,0,0,0)
            )
        e_u=1+self.epsilon(
            self.f(1,1,0,0),\
            self.f(self.h-2,0,0,0),\
            self.f(self.h-1,1,0,0)
            )

        cord_weight_list = \
                [ e_root ]+\
                [ e_u ]+\
                [
                (1+self.epsilon(
                    self.f(2,0,0,0),\
                    self.f(self.h-k-1,0,0,0),\
                    self.f(self.h-k+1,0,0,0))
                    )
            for k in range(2,self.h)
            ]

        numerators=[e_root_norm]+[((self.c-1)*e_u+e_r*e_u+e_r)/e_r]+[self.c+cord_weight_list[k] for k in range(2,self.h)]

        eq_ratio_list = [ self.root_cluster_eq_ratio() ] + [self.sub_root_cluster_eq_ratio()] +[ self.eq_ratio(self.h-k) for k in range(2,self.h) ]

        out = np.sum( [np.sum([eq_ratio_list[k]*numerators[k]/np.prod(cord_weight_list[k:i]) for k in range(i)]) for i in range(1,self.h+1)] )
        return out

class MF_patternWalker_general(rw.fullProbPatternWalker):
    """
    reproduces the methods f, epsilon
    of overlap_MF_patternWalker but without assuming that
    the tree is a c-ary tree.
    Method sets all edge weights to their predicted mean value
    """
    def __init__(self,*args,**params):
        super(MF_patternWalker_general,self).__init__(*args,**params)
        self.set_weights()
        self.set_coordinates()
        self.c = self.degree(self.root)
        if self.overlap>self.pattern_len*(self.c-1)/(2*self.c):
            self.overlap=(self.pattern_len-int(self.pattern_len/self.c))/2
        self.O_list=np.array([
            max(0,int(self.pattern_len/self.c)*(2-i)+2*self.overlap)+\
            max(0,i*int(self.pattern_len/self.c)-self.pattern_len+2*self.overlap)
            for i in range(2,self.c+1)])
        self.U_list=np.array([
            max(0,-int(self.pattern_len/self.c)*(2-i)-2*self.overlap)+\
            max(0,-i*int(self.pattern_len/self.c)+self.pattern_len-2*self.overlap)
            for i in range(2,self.c+1)])
        self.O_hh=np.sum(self.O_list)/(self.pattern_len*(self.c-1))
        self.O_ll=np.sum(self.U_list)/(self.pattern_len*(self.c-1))
        self.O_hl=(1-self.O_hh-self.O_ll)/2
        self.O_lh=self.O_hl

        for node in self.nodes:
            self.set_mean_weights(node)

    def f(self,k,up,down,m,mu=2,**kwargs):
        a_h=self.a_high
        a_l=self.a_low
        a=self.a_root
        Gamma=self.Gamma
        Gammap=self.Gamma_root

        out=0.
        coordinates=(k,up,down,m)

        if up==0 and down+m==0:
            #target branch
            f_h=MF_patternWalker.f(self,*coordinates,ajl=a_h,ajr=a_h)
            f_l=MF_patternWalker.f(self,*coordinates,ajl=a_l,ajr=a_l)
            out=f_h*(self.pattern_len/self.c+2*self.overlap)/self.pattern_len+\
            f_l*(
                self.pattern_len-self.pattern_len/self.c-2*self.overlap
                )/self.pattern_len

        elif up==1 and down==0:
            #target branch up to root
            f_h=MF_patternWalker.f(self,*coordinates,ajl=a_h,ajr=a)
            f_l=MF_patternWalker.f(self,*coordinates,ajl=a_l,ajr=a)
            out=f_h*(self.pattern_len/self.c+2*self.overlap)/self.pattern_len+\
                f_l*(
                    self.pattern_len-self.pattern_len/self.c-2*self.overlap
                    )/self.pattern_len

        elif up==0 and down==1:
            #root down to non-target branch
            f_h=MF_patternWalker.f(self,*coordinates,ajl=a,ajr=a_h)
            f_l=MF_patternWalker.f(self,*coordinates,ajl=a,ajr=a_l)
            out=f_h*(self.pattern_len/self.c+2*self.overlap)/self.pattern_len+\
                f_l*(
                    self.pattern_len-self.pattern_len/self.c-2*self.overlap
                    )/self.pattern_len

        elif up+k==0 and down==0:
            #non-targget branch
            f_h=MF_patternWalker.f(self,*coordinates,ajl=a_h,ajr=a_h)
            f_l=MF_patternWalker.f(self,*coordinates,ajl=a_l,ajr=a_l)
            out=f_h*(self.pattern_len/self.c+2*self.overlap)/self.pattern_len+\
                f_l*(
                    self.pattern_len-self.pattern_len/self.c-2*self.overlap
                    )/self.pattern_len

        else:
            #from target branch over root to non-target branch
            f_hh=MF_patternWalker.f(self,*coordinates,ajl=a_h,ajr=a_h)
            f_hl=MF_patternWalker.f(self,*coordinates,ajl=a_h,ajr=a_l)
            f_lh=MF_patternWalker.f(self,*coordinates,ajl=a_l,ajr=a_h)
            f_ll=MF_patternWalker.f(self,*coordinates,ajl=a_l,ajr=a_l)
            if mu==0:
                #is this the case where we don't care?
                out=self.O_hh*f_hh+self.O_lh*f_lh+self.O_hl*f_hl+self.O_ll*f_ll
            else:
                out=self.O_list[mu-2]/self.pattern_len*f_hh+\
                    (self.pattern_len-self.U_list[mu-2]-self.O_list[mu-2])/self.pattern_len*\
                    (f_hl+f_lh)/2+self.U_list[mu-2]/self.pattern_len*f_ll

        return out

    def epsilon(self,f2,fk,fk2):
        # NOTE:  in the notation of the paper this is epsilon-1, to avoid
        # too many distinct cases
        L=self.pattern_len
        out=0.
        if fk==0:
            if f2==0:
                out=0.
            else:
                out=f2*self.pattern_len
        else:
            # f2=f2/fk2
            dk1_inv_exp=(1-(1-fk)**(self.pattern_len+1))/((self.pattern_len+1)*fk)
            out=-1+(1+L*f2*fk2/(1-fk)-fk2*(1-fk-f2)/(fk*(1-fk)))*dk1_inv_exp+fk2*(1-fk-f2)/(fk*(1-fk))

        return out

    def set_mean_weights(self,node):

        neigh=list(self.neighbors(node))
        out={}
        node_coordinates = list(self.nodes[node]['coordinates'])
        try:
            toward_target = nx.shortest_path(self,node,self.target_node)[1] # path to target
        except IndexError:
            pass

        if len(neigh) == 1:
            out={ (node,*neigh):1. }

        elif node==self.root:
            c = len(neigh)
            h = node_coordinates[0]+1
            weights=[(1+\
                self.epsilon(
                    self.f(0,1,1,0,mu),\
                    self.f(h-1,0,0,0,mu),\
                    self.f(h-1,1,1,0,mu))
                )
                for mu in range(2,c+1)
                ]
            e_0=np.prod(weights)
            normalisation=1/(e_0+\
                np.sum([ e_0/weight for weight in weights ])
                )
            #counter-clockwise. First towards target, then the other parts in order
            out[(node,toward_target)]=e_0*normalisation
            for neighbor in set(neigh)-set([toward_target]):
                part=self.nodes[neighbor]['coordinates'][4]
                out[(node,neighbor)]=e_0*normalisation/weights[part-2]


        elif self.root in neigh and node_coordinates[2]==0:
            c=len(neigh)-1
            h=node_coordinates[0]+1
            e_r=1+self.epsilon(self.f(2,0,0,0),self.f(h-2,0,0,0),self.f(h,0,0,0))
            e_u=1+self.epsilon(self.f(1,1,0,0),self.f(h-2,0,0,0),self.f(h-1,1,0,0))
            normalisation=1/(e_u*(c-1)+e_r+e_r*e_u)
            for neighbor in neigh:
                #counter-clockwise, starting from the target, then other children,
                #finally root
                if neighbor ==toward_target:
                    out[(node,neighbor)]=e_u*e_r*normalisation
                elif not neighbor==self.root:
                    out[(node,neighbor)]=e_u*normalisation
                else:
                    out[(node,neighbor)]=e_r*normalisation

        elif self.root in neigh:
            node_coordinates[2]=0 # easier access to targetwards and opposite neighbour
            c = len(neigh)-1
            h = node_coordinates[0]+1
            connector = [0,0,1,1] # connection between targetwards and "opposite" neighbour
            e=1+self.epsilon(self.f(*connector,node_coordinates[-1]),\
            self.f(h-1,1,0,0,node_coordinates[4]),\
            self.f(h-1,1,1,1,node_coordinates[-1]
            ))
            for neighbor in neigh:
                #counter-clockwise. first root (=targetwards), then children
                if neighbor==self.root:
                        out[(node,neighbor)]=e/(c+e)
                else:
                    out[ (node,neighbor)]=1/(c+e)

        else:
            c=len(neigh)-1
            connector=[2,0,0,0] # connection between targetwards and "opposite" neighbour
            if node_coordinates[3]>0:
                ## TODO: check back to see if this is important
                node_coordinates[3]-=1 # easier access to targetwards and opposite neighbour
                connector=[0,0,0,2]
            else:
                node_coordinates[0]-=1 # easier access to targetwards and opposite neighbour
            e=1+self.epsilon(self.f(*connector,node_coordinates[-1]),\
                self.f( *node_coordinates ),
                self.f( *[node_coordinates[i]+connector[i] for i in range(4)]
                ))

            for neighbor in neigh:
                #counter-clockwise. first targetwards, then children
                if neighbor==toward_target:
                    out[( node,neighbor)] = e/(c+e)
                else:
                    out[( node,neighbor )] = 1/(c+e)
        nx.set_edge_attributes(self,out,'mean_weight')


class MFPropertiesCAryTrees(overlap_MF_patternWalker):
    """
    Hosts some of the methods from the overlap_MF_patternwalker class that
    function without a graph. Results are for c-ary trees of height h and
    offspring number c.
    """
    def __init__(self,c,h,pattern_len,a_root,a_low,\
        a_high,overlap,Gamma,Gamma_root):
        self.c=c
        self.h=h
        self.a_root=a_root
        if a_high<=(1-a_root)*Gamma_root+a_root and a_high>=(1-a_root)*Gamma_root:
            self.a_high=a_high
        else:
            self.a_high=(1-a_root)*Gamma_root+a_root
        if a_low>=(1-a_root)*Gamma_root and a_low<=(1-a_root)*Gamma_root+a_root:
            self.a_low=a_low
        else:
            self.a_low=(1-a_root)*Gamma_root+a_root/10
        self.overlap=overlap
        self.Gamma_root=Gamma_root
        self.num_parts=c
        self.part_size=int(pattern_len/self.num_parts)
        self.pattern_len=pattern_len
        self.Gamma=Gamma

        if self.overlap>self.pattern_len*(self.c-1)/(2*self.c):
            self.overlap=(self.pattern_len-int(self.pattern_len/self.c))/2
        self.O_list=np.array([
            max(0,int(self.pattern_len/self.c)*(2-i)+2*self.overlap)+\
            max(0,i*int(self.pattern_len/self.c)-self.pattern_len+2*self.overlap)
            for i in range(2,c+1)])
        self.U_list=np.array([
            max(0,-int(self.pattern_len/self.c)*(2-i)-2*self.overlap)+\
            max(0,-i*int(self.pattern_len/self.c)+self.pattern_len-2*self.overlap)
            for i in range(2,c+1)])
        self.O_hh=np.sum(self.O_list)/(self.pattern_len*(self.c-1))
        self.O_ll=np.sum(self.U_list)/(self.pattern_len*(self.c-1))
        self.O_hl=(1-self.O_hh-self.O_ll)/2
        self.O_lh=self.O_hl


    def f(self,k,up,down,m,mu=2,**kwargs):
        a_h=self.a_high
        a_l=self.a_low
        a=self.a_root
        Gamma=self.Gamma
        Gammap=self.Gamma_root

        out=0.
        coordinates=(k,up,down,m)

        if up==0 and down+m==0:
            #target branch
            f_h=MF_patternWalker.f(self,*coordinates,ajl=a_h,ajr=a_h)
            f_l=MF_patternWalker.f(self,*coordinates,ajl=a_l,ajr=a_l)
            out=f_h*(self.pattern_len/self.c+2*self.overlap)/self.pattern_len+\
            f_l*(
                self.pattern_len-self.pattern_len/self.c-2*self.overlap
                )/self.pattern_len

        elif up==1 and down==0:
            #target branch up to root
            f_h=MF_patternWalker.f(self,*coordinates,ajl=a_h,ajr=a)
            f_l=MF_patternWalker.f(self,*coordinates,ajl=a_l,ajr=a)
            out=f_h*(self.pattern_len/self.c+2*self.overlap)/self.pattern_len+\
                f_l*(
                    self.pattern_len-self.pattern_len/self.c-2*self.overlap
                    )/self.pattern_len

        elif up==0 and down==1:
            #root down to non-target branch
            f_h=MF_patternWalker.f(self,*coordinates,ajl=a,ajr=a_h)
            f_l=MF_patternWalker.f(self,*coordinates,ajl=a,ajr=a_l)
            out=f_h*(self.pattern_len/self.c+2*self.overlap)/self.pattern_len+\
                f_l*(
                    self.pattern_len-self.pattern_len/self.c-2*self.overlap
                    )/self.pattern_len

        elif up+k==0 and down==0:
            #non-targget branch
            f_h=MF_patternWalker.f(self,*coordinates,ajl=a_h,ajr=a_h)
            f_l=MF_patternWalker.f(self,*coordinates,ajl=a_l,ajr=a_l)
            out=f_h*(self.pattern_len/self.c+2*self.overlap)/self.pattern_len+\
                f_l*(
                    self.pattern_len-self.pattern_len/self.c-2*self.overlap
                    )/self.pattern_len

        else:
            #from target branch over root to non-target branch
            f_hh=MF_patternWalker.f(self,*coordinates,ajl=a_h,ajr=a_h)
            f_hl=MF_patternWalker.f(self,*coordinates,ajl=a_h,ajr=a_l)
            f_lh=MF_patternWalker.f(self,*coordinates,ajl=a_l,ajr=a_h)
            f_ll=MF_patternWalker.f(self,*coordinates,ajl=a_l,ajr=a_l)
            if mu==0:
                #is this the case where we don't care?
                out=self.O_hh*f_hh+self.O_lh*f_lh+self.O_hl*f_hl+self.O_ll*f_ll
            else:
                out=self.O_list[mu-2]/self.pattern_len*f_hh+\
                    (self.pattern_len-self.U_list[mu-2]-self.O_list[mu-2])/self.pattern_len*\
                    (f_hl+f_lh)/2+self.U_list[mu-2]/self.pattern_len*f_ll

        return out

    def root_cluster_eq_ratio(self):
        #eq prob of cluster divided by eq prob of articulation pt, here the root itself

        branch_weights=[(1+\
            self.epsilon(
                self.f(0,1,1,0,mu),\
                self.f(self.h-1,0,0,0,mu),\
                self.f(self.h-1,1,1,0,mu))
            )
            for mu in range(2,self.c+1)
            ]

        e_0=np.prod(branch_weights)
        e_r_list=[e_0/weight for weight in branch_weights]
        branch_normalisation=1/(e_0+np.sum(e_r_list))
        bias_dict={
            mu: [e_0,e_r_list[mu-2], 1+self.epsilon(self.f(0,0,1,1,mu),\
                self.f(self.h-1,1,0,0,mu),self.f(self.h-1,1,1,1,mu))]\
                +[
                    1+self.epsilon(
                        self.f(0,0,0,2,mu),\
                        self.f(self.h-1,1,1,l,mu),\
                        self.f(self.h-1,1,1,l+2,mu)
                    )
                for l in range(self.h-2)
                ]
            for mu in range(2,self.c+1)
            }

        out=1+np.sum([branch_normalisation*bias_dict[mu][1]*\
                (
                np.sum([
                    self.c**l*(self.c+bias_dict[mu][l+2])/\
                    np.prod([bias_dict[mu][k+2] for k in range(l+1)])
                for l in range(self.h-1)
                ])+\
                self.c**(self.h-1)/\
                    np.prod([bias_dict[mu][l+2] for l in range(self.h-1)])
                )
            for mu in range(2,self.c+1)
            ])

        return out

    def MF_mfpt(self,ajl=None,ajr=None,a=None,Gamma=None,Gammap=None,**kwargs):
        #just under the root we need to distinguish two different paths, namely,
        # the one going over the root and the ones staying in the target-part

        branch_weights=[(1+\
            self.epsilon(
                self.f(0,1,1,0,mu),\
                self.f(self.h-1,0,0,0,mu),\
                self.f(self.h-1,1,1,0,mu))
            )
            for mu in range(2,self.c+1)
            ]
        e_root=np.prod(branch_weights)

        e_root_norm=e_root+np.sum([
                e_root/weight for weight in branch_weights])

        e_r=1+self.epsilon(
            self.f(2,0,0,0),\
            self.f(self.h-2,0,0,0),\
            self.f(self.h,0,0,0)
            )
        e_u=1+self.epsilon(
            self.f(1,1,0,0),\
            self.f(self.h-2,0,0,0),\
            self.f(self.h-1,1,0,0)
            )

        cord_weight_list = \
                [ e_root ]+\
                [ e_u ]+\
                [
                (1+self.epsilon(
                    self.f(2,0,0,0),\
                    self.f(self.h-k-1,0,0,0),\
                    self.f(self.h-k+1,0,0,0))
                    )
            for k in range(2,self.h)
            ]

        numerators=[e_root_norm]+[((self.c-1)*e_u+e_r*e_u+e_r)/e_r]+[self.c+cord_weight_list[k] for k in range(2,self.h)]

        eq_ratio_list = [ self.root_cluster_eq_ratio() ] + [self.sub_root_cluster_eq_ratio()] +[ self.eq_ratio(self.h-k) for k in range(2,self.h) ]

        out = np.sum( [ np.sum([eq_ratio_list[k]*numerators[k]/np.prod(cord_weight_list[k:i]) for k in range(i)]) for i in range(1,self.h+1)] )
        return out

def MF_mfpt_cary_tree(*args):
    """
    wrapper for MFPropertiesCAryTrees.MF_mfpt
    """
    #c,h,pattern_len,a_root,a_low,\
    # a_high,overlap,Gamma,Gamma_root
    G = MFPropertiesCAryTrees(*args)
    return G.MF_mfpt()

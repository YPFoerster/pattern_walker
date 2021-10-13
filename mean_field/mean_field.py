"""
Class for walker with approximate average-patterns, in particular the
approximate MFPT from root to target.
"""

import pattern_walker as rw
import numpy as np
import networkx as nx

__all__ = [
    'MF_patternWalker', 'overlap_MF_patternWalker'
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
        if aj is None:
            aj=self.high_child_prior
        Gamma=self.flip_rate

        if aj>0:
            out= np.array(
                    [[1-aj*(1-(1-Gamma)**k),aj*(1-(1-Gamma)**k)],
                             [(1-aj)*(1-(1-Gamma)**k),1-(1-aj)*(1-(1-Gamma)**k)]]
                    )
        elif aj==0:
            out=np.linalg.matrix_power(np.array( [[1,0],[Gamma,1-Gamma]] ),k)
        return out

    def R_up(self,aj=None,a=None,Gammap=None):
        if aj is None:
            ajl=self.high_child_prior
        if a is None:
            a=self.root_prior
        if Gammap is None:
            Gammap=self.root_flip_rate

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
        if aj is None:
            aj=self.high_child_prior
        if a is None:
            a=self.root_prior
        if Gammap is None:
            Gammap=self.root_flip_rate

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

    #for full overlap, the pattern distance rates at graph distance k from the target. Cases separated depending
    #whether the root needs to be involved
    #k includes the upward root link if applicable, m includes the downwards root link, if applicable
    #so the sum of k and m has to be the graph distance

    def f(self,k,up=0,down=0,m=0,mu=2,ajl=None,ajr=None):
        if ajl is None:
            ajl=self.high_child_prior
        if ajr is None:
            ajr=self.high_child_prior
        a=self.root_prior
        Gamma=self.flip_rate
        Gammap=self.root_flip_rate
        out=0.

        if k>0 and up+down+m==0:
            out=self.Q_power(k,aj=ajl)
            out = ajl*out[1,0]+(1-ajl)*out[0,1]

        elif up==1 and down+m==0:
            out=self.Q_power(k,aj=ajl).dot(self.R_up(aj=ajl))
            out = ajl*out[1,0]+(1-ajl)*out[0,1]

        elif up==down==1:
            out=self.Q_power(k,aj=ajl).dot(self.R_up(aj=ajl)).dot(self.R_down(aj=ajr)).dot(self.Q_power(m,aj=ajr))
            out = ajl*out[1,0]+(1-ajl)*out[0,1]

        elif up==0 and down==1:
            out=self.R_down(aj=ajr).dot(self.Q_power(m,aj=ajr))
            out = a*out[1,0]+(1-a)*out[0,1]

        elif down==0:
            out=self.Q_power(m,aj=ajr)
            out = ajr*out[1,0]+(1-ajr)*out[0,1]

        return out

    def weight_bias(self,f2,fk,fk2):
        #of E(w_l/w_r)=1+epsilon return epsilon
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
        ajl=self.high_child_prior
        ajr=self.high_child_prior
        a=self.root_prior
        Gamma=self.flip_rate
        Gammap=self.root_flip_rate
        kwargs={'c':self.c,'h':self.h,'L':self.pattern_len,'ajl':ajl,'ajr':ajr, 'a':a , 'Gamma':Gamma,'Gammap':Gammap}
        #eq prob of cluster divided by eq prob of articulation pt, here the root itself
        bias_list=[ self.weight_bias( self.f(0,1,1,0),self.f(self.h-1,0,0,0),self.f(self.h-1,1,1,0) ), self.weight_bias(self.f(0,0,1,1,ajl=a),self.f(self.h-1,1,0,0,ajr=a),self.f(self.h-1,1,1,1,ajr=a)) ]+[ self.weight_bias( self.f(0,0,0,2),self.f(self.h-1,1,1,m),self.f(self.h-1,1,3,m) )  for m in range(self.h-1) ]
        out=1+ (self.c-1)/(self.c+bias_list[0])*\
            (
            np.sum([
                    self.c**(l-1)*(self.c+1+bias_list[l])/np.prod( [1+bias_list[k] for k in range(1,l+1)])
                    for l in range(1,self.h) ])+\
            self.c**(self.h-1)/np.prod( [1+bias_list[k] for k in range(1,self.h)])
            )
        return out

    def sub_root_cluster_eq_ratio(self):
        #eq prob of cluster divided by eq prob of articulation pt, here the node under the root
        #just under the root things are a bit messy, hence the following epsilons
        ajl=self.high_child_prior
        ajr=self.high_child_prior
        a=self.root_prior
        Gamma=self.flip_rate
        Gammap=self.root_flip_rate
        kwargs={'c':self.c,'h':self.h,'L':self.pattern_len,'ajl':ajl,'ajr':ajr, 'a':a , 'Gamma':Gamma,'Gammap':Gammap}

        e_r=self.weight_bias(self.f(2,0,0,0),self.f(self.h-2,0,0,0),self.f(self.h,0,0,0))
        e_u=self.weight_bias(self.f(1,1,0,0,ajr=a),self.f(self.h-2,0,0,0),self.f(self.h-1,1,0,0))

        bias_list=[ e_r,e_u ]+[ self.weight_bias( self.f(2,0,0,0),self.f(self.h-1+m,0,0,0),self.f(self.h+1+m,0,0,0) )  for m in range(self.h-2) ]
        out=1+(self.c-1)*(1+e_u)/( (self.c+bias_list[0])*(1+bias_list[1])+1+bias_list[0] )*\
            (
            np.sum([
                        self.c**(l-1)*(self.c+1+bias_list[l+1])/np.prod( [1+bias_list[k] for k in range(2,l+2)])
                for l in range(1,self.h-1) ])+\
            self.c**(self.h-2)/np.prod( [1+bias_list[k] for k in range(2,self.h)])
            )
        return out

    def eq_ratio(self,k):
        #eq prob of cluster divided by eq prob of articulation pt at height k over target
        ajl=self.high_child_prior
        ajr=self.high_child_prior
        a=self.root_prior
        Gamma=self.flip_rate
        Gammap=self.root_flip_rate
        kwargs={'c':self.c,'h':self.h,'L':self.pattern_len,'ajl':ajl,'ajr':ajr, 'a':a , 'Gamma':Gamma,'Gammap':Gammap}

        if k==0:
            return 1
        bias_list=[ self.weight_bias( self.f(2,0,0,0),self.f(l,0,0,0),self.f(l+2,0,0,0) )  for l in range(k-1,2*k-1) ]
        out=1+ (self.c-1)/(self.c+1+bias_list[0])*\
            (
            np.sum([
                        self.c**(l-1)*(self.c+1+bias_list[l])/np.prod( [ 1+bias_list[m] for m in range(1,l+1)])
                for l in range(1,k) ])+\
            self.c**(k-1)/np.prod( [ 1+bias_list[l] for l in range(1,k)])
            )
        return out

    def MF_mfpt(self,ajl=None,ajr=None,a=None,Gamma=None,Gammap=None,**kwargs):
        ajl=self.high_child_prior
        ajr=self.high_child_prior
        a=self.root_prior
        Gamma=self.flip_rate
        Gammap=self.root_flip_rate
        kwargs={'c':self.c,'h':self.h,'L':self.pattern_len,'ajl':ajl,'ajr':ajr, 'a':a , 'Gamma':Gamma,'Gammap':Gammap}
        #just under the root things are a bit messy, hence the following epsilons
        e_r=self.weight_bias(self.f(2,0,0,0),self.f(self.h-2,0,0,0),self.f(self.h,0,0,0))
        e_u=self.weight_bias(self.f(1,1,0,0,ajr=a),self.f(self.h-2,0,0,0),self.f(self.h-1,1,0,0,ajr=a))
        cord_weight_list = [ (1+self.weight_bias(self.f(2,0,0,0),self.f(k,0,0,0),self.f(k+2,0,0,0)))/(self.c+1+self.weight_bias(self.f(2,0,0,0),self.f(k,0,0,0),self.f(k+2,0,0,0))) for k in range(0,self.h-2) ] + [ (1+e_r)*(1+e_u)/( (self.c+e_r)*(1+e_u)+1+e_u ) ] + [ (1+self.weight_bias(self.f(0,1,1,0),self.f(self.h-1,0,0,0),self.f(self.h-1,1,1,0)))/(self.c+self.weight_bias(self.f(0,1,1,0),self.f(self.h-1,0,0,0),self.f(self.h-1,1,1,0))) ]
        eq_ratio_list = [ self.eq_ratio(k) for k in range(1,self.h-1) ]+[self.sub_root_cluster_eq_ratio()]+[ self.root_cluster_eq_ratio() ]
        out = np.sum(np.sum( [[eq_ratio_list[k]/cord_weight_list[k] for k in range(i,self.h)] for i in range(self.h)] ))
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
        if nodelist is None:
            nodelist=[self.root]+list( self.nodes -set([self.root,self.target_node])  )+[self.target_node]
        out=nx.DiGraph()
        out.add_nodes_from(self.nodes)
        for node in self.nodes:
            weights=self.mean_out_weights(node)
            out.add_weighted_edges_from(weights)
        return out,nx.to_numpy_array(out,nodelist=nodelist)

    def mean_out_weights(self,node):
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
                self.weight_bias(
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
            out.append((node,toward_target,e_0*normalisation))
            for neighbor in set(neigh)-set([toward_target]):
                part=self.nodes[neighbor]['coordinates'][4]
                out.append((node,neighbor,e_0*normalisation/weights[part-2]))


        elif self.root in neigh and self.nodes[node]['coordinates'][2]==0:
            e_r=1+self.weight_bias(self.f(2,0,0,0),self.f(self.h-2,0,0,0),self.f(self.h,0,0,0))
            e_u=1+self.weight_bias(self.f(1,1,0,0),self.f(self.h-2,0,0,0),self.f(self.h-1,1,0,0))
            normalisation=1/(e_u*(self.c-1)+e_r+e_r*e_u)
            for neighbor in neigh:
                if neighbor==self.root:
                    out.append( (node,neighbor,e_r*normalisation ) )
                elif neighbor ==toward_target:
                    out.append( (node,neighbor,e_u*e_r*normalisation ) )
                else:
                    out.append( (node,neighbor,e_u*normalisation ) )

        elif self.root in neigh:
            coordinates=list(self.nodes[node]['coordinates'])
            coordinates[2]=0
            e=1+self.weight_bias(self.f(0,0,1,1,coordinates[4]),self.f(self.h-1,1,0,0,coordinates[4]),self.f(self.h-1,1,1,1,coordinates[4]))
            for neighbor in neigh:
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
            e=1+self.weight_bias(self.f(*short_path,coordinates[-1]),self.f( *coordinates ),self.f( *[coordinates[i]+short_path[i] for i in range(4)] ))
            # TODO: following line with tightness 0/1 suitable for unit test
            #print('e:',e,self.nodes[node]['coordinates'],self.f(*coordinates))
            for neighbor in neigh:
                if neighbor==toward_target:
                    out.append(( node,neighbor,e/(self.c+e)))
                else:
                    out.append(( node,neighbor,1/(self.c+e) ))
        return out


class overlap_MF_patternWalker(MF_patternWalker):
    #does all of the above with the correct parameters as given by G
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
        a_h=self.high_child_prior
        a_l=self.low_child_prior
        a=self.root_prior
        Gamma=self.flip_rate
        Gammap=self.root_flip_rate

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
        ajl=self.high_child_prior
        ajr=self.high_child_prior
        a=self.root_prior
        Gamma=self.flip_rate
        Gammap=self.root_flip_rate
        kwargs={'c':self.c,'h':self.h,'L':self.pattern_len,'ajl':ajl,'ajr':ajr,\
            'a':a , 'Gamma':Gamma,'Gammap':Gammap}
        #eq prob of cluster divided by eq prob of articulation pt, here the root itself

        ## TODO: this can be simplified in the way done for the full-overlap case
        e_0=np.prod([
            (1+self.weight_bias(self.f(0,1,1,0,mu),self.f(self.h-1,0,0,0,mu),\
                self.f(self.h-1,1,1,0,mu)))
            for mu in range(2,self.c+1)
            ])
        normalisation=1/(e_0+np.sum([
            e_0/(1+self.weight_bias(self.f(0,1,1,0,mu),self.f(self.h-1,0,0,0,mu),\
                self.f(self.h-1,1,1,0,mu)))
            for mu in range(2,self.c+1)
            ]))
        e_r_list=[
            e_0/(1+self.weight_bias(self.f(0,1,1,0,mu),self.f(self.h-1,0,0,0,mu),\
                self.f(self.h-1,1,1,0,mu)))
            for mu in range(2,self.c+1)
            ]
        bias_dict={
            mu: [e_r_list[mu-2], self.weight_bias(self.f(0,0,1,1,mu,ajl=a),\
                self.f(self.h-1,1,0,0,mu,ajr=a),self.f(self.h-1,1,1,1,mu,ajr=a))]\
                +[
                    self.weight_bias(
                        self.f(0,0,0,2,mu),\
                        self.f(self.h-1,1,1,m,mu),\
                        self.f(self.h-1,1,1,m+2,mu)
                    )
                for m in range(self.h-1)
                ]
            for mu in range(2,self.c+1)
            }
        out=1+np.sum([
                bias_dict[mu][0]*normalisation*\
                (
                np.sum([
                    self.c**(l-1)*(self.c+1+bias_dict[mu][l])/\
                        np.prod([1+bias_dict[mu][k] for k in range(1,l+1)])
                for l in range(1,self.h)
                ])+\
                self.c**(self.h-1)/\
                    np.prod([1+bias_dict[mu][k] for k in range(1,self.h)])
                )
            for mu in range(2,self.c+1)
            ])

        return out

    def MF_mfpt(self,ajl=None,ajr=None,a=None,Gamma=None,Gammap=None,**kwargs):
        ajl=self.high_child_prior
        ajr=self.high_child_prior
        a=self.root_prior
        Gamma=self.flip_rate
        Gammap=self.root_flip_rate
        kwargs={'c':self.c,'h':self.h,'L':self.pattern_len,'ajl':ajl,'ajr':ajr, 'a':a , 'Gamma':Gamma,'Gammap':Gammap}
        #just under the root things are a bit messy, hence the following epsilons

        ## TODO: simplify,spot errors
        e_root=np.prod([
                (1+self.weight_bias(
                    self.f(0,1,1,0,mu),\
                    self.f(self.h-1,0,0,0,mu),\
                    self.f(self.h-1,1,1,0,mu)
                    )
                )
            for mu in range(2,self.c+1)
            ])
        e_root=e_root/(e_root+np.sum([
                e_root/(1+self.weight_bias(
                    self.f(0,1,1,0,mu),\
                    self.f(self.h-1,0,0,0,mu),\
                    self.f(self.h-1,1,1,0,mu)
                    )
                )
            for mu in range(2,self.c+1)
            ]))

        e_r=1+self.weight_bias(
            self.f(2,0,0,0),\
            self.f(self.h-2,0,0,0),\
            self.f(self.h,0,0,0)
            )
        e_u=1+self.weight_bias(
            self.f(1,1,0,0,ajr=a),\
            self.f(self.h-2,0,0,0),\
            self.f(self.h-1,1,0,0,ajr=a)
            )
        cord_weight_list = [
                (1+self.weight_bias(
                    self.f(2,0,0,0),\
                    self.f(k,0,0,0),\
                    self.f(k+2,0,0,0))
                    )/(self.c+1+self.weight_bias(
                        self.f(2,0,0,0),\
                        self.f(k,0,0,0),\
                        self.f(k+2,0,0,0))
                        )
            for k in range(0,self.h-2)
            ] + \
            [ (e_r)*(e_u)/( (self.c-1+e_r)*(e_u)+e_u ) ] +\
            [ e_root ]
        eq_ratio_list = [ self.eq_ratio(k) for k in range(1,self.h-1) ]+\
            [self.sub_root_cluster_eq_ratio()]+\
            [ self.root_cluster_eq_ratio() ]
        out = np.sum(np.sum([
                [eq_ratio_list[k]/cord_weight_list[k] for k in range(i,self.h)]
            for i in range(self.h)
            ] ))

        return out

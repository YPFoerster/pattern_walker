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

    def Q_power(self,k,aj=None):
        if aj is None:
            aj=self.high_child_prior
        Gamma=self.flip_rate

        if aj>0:
#             out= np.array(
#                     [[1-ajl*(1-(1-Gamma/ajl)**k),ajl*(1-(1-Gamma/ajl)**k)],
#                              [(1-ajl)*(1-(1-Gamma/ajl)**k),1-(1-ajl)*(1-(1-Gamma/ajl)**k)]]
#                     )
            out= np.array(
                    [[1-aj*(1-(1-Gamma)**k),aj*(1-(1-Gamma)**k)],
                             [(1-aj)*(1-(1-Gamma)**k),1-(1-aj)*(1-(1-Gamma)**k)]]
                    )
        elif aj==0:
#             out=np.linalg.matrix_power(np.array( [[1-Gamma,Gamma],[Gamma,1-Gamma]] ),k)
            out=np.linalg.matrix_power(np.array( [[1,0],[Gamma,1-Gamma]] ),k)
        return out

    def Qp_up(self,aj=None,a=None,Gammap=None):
        if aj is None:
            ajl=self.high_child_prior
        if a is None:
            a=self.root_prior
        if Gammap is None:
            Gammap=self.root_flip_rate

        if aj>0:
            out= np.array(
                    [[(1-a)*(1-Gammap)/(1-aj),(a-aj+(1-a)*Gammap)/(1-aj)],
                     [(1-a)/aj*Gammap,1-(1-a)/aj*Gammap]]
                    )
        elif aj==0:
            out=np.array(
                    [[1-Gammap,Gammap],
                     [Gammap,1-Gammap]]
                    )
        return out

    def Qp_down(self,aj=None,a=None,Gammap=None):
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
                     [Gammap,1-Gammap]]
                    )
        return out

    #for full overlap, the pattern distance rates at graph distance k from the target. Cases separated depending
    #whether the root needs to be involved
    #k includes the upward root link if applicable, m includes the downwards root link, if applicable
    #so the sum of k and m has to be the graph distance

    def kappa(self,k,ajl=None,Gamma=None):
        if ajl is None:
            ajl=self.high_child_prior
        if Gamma is None:
            Gamma=self.flip_rate
        out=(1-(1-Gamma)**k)
        return out

    def f_left(self,k,ajl=None,ajr=None,a=None,Gamma=None,Gammap=None,**kwargs):
        if ajl is None:
            ajl=self.high_child_prior
        if Gamma is None:
            Gamma=self.flip_rate

        #no root involved
        #checked and found to be correct
        temp=2*ajl*(1-ajl)*(1-(1-Gamma)**k)
        out=self.Q_power(k,ajl,Gamma)
        out=ajl*out[1,0]+(1-ajl)*out[0,1]
        #print('left:',temp-out)
        return out


    def f_up(self,k,ajl=None,ajr=None,a=None,Gamma=None,Gammap=None,**kwargs):
        if ajl is None:
            ajl=self.high_child_prior
        if a is None:
            a=self.root_prior
        if Gamma is None:
            Gamma=self.flip_rate
        if Gammap is None:
            Gammap=self.root_flip_rate

        #last edge goes up to the root
        #checked and found to be correct
        temp= 2*(1-a)*Gammap+a-ajl+2*(1-(1-Gamma)**(k))*(1-a)*ajl*(1-Gammap/ajl)
        out=self.Q_power(k,ajl,Gamma).dot(self.Qp_up(ajl,a,Gammap))
        out=ajl*out[1,0]+(1-ajl)*out[0,1]
        #print('up:',temp-out)
        return out



    def f_up_down(self,k,m=1,ajl=None,ajr=None,a=None,Gamma=None,Gammap=None,**kwargs):
        if ajl is None:
            ajl=self.high_child_prior
        if ajr is None:
            ajr=self.high_child_prior
        if a is None:
            a=self.root_prior
        if Gamma is None:
            Gamma=self.flip_rate
        if Gammap is None:
            Gammap=self.root_flip_rate

        #tested and seems correct
        temp=1-(1-a)*((1-Gammap)**2+Gammap**2)-\
            1/a*(ajl-(1-a)*Gammap)*(ajr-(1-a)*Gammap)-(a-ajr+(1-a)*Gammap)*(1-ajl-(1-a)*(1-Gammap))/a+\
            2*self.kappa(k,ajl)*(1-a)*(ajl-Gammap)*(ajr-Gammap)/a+\
            2*self.kappa(m,ajr)*(1-self.kappa(k,ajl))*(1-a)*(ajl-Gammap)*(ajr-Gammap)/a
        #last two edges go over the root, up and then down from leftmost to some branch on the right
        out=self.Q_power(k,ajl,Gamma).dot(self.Qp_up(ajl,a,Gammap).dot(self.Qp_down(ajr,a,Gammap).dot(self.Q_power(m,ajr,Gamma))))
        out=ajl*out[1,0]+(1-ajl)*out[0,1]
        #print('fud:',out-temp,out)
        return out

    def f_down(self,m,ajl=None,ajr=None,a=None,Gamma=None,Gammap=None,**kwargs):
        if ajr is None:
            ajr=self.high_child_prior
        if a is None:
            a=self.root_prior
        if Gamma is None:
            Gamma=self.flip_rate
        if Gammap is None:
            Gammap=self.root_flip_rate
        #seems correct
        temp= 2*(1-a)*Gammap+a-ajr+2*self.kappa(m,ajr,Gamma)*(1-a)*(ajr-Gammap)
        out=self.Qp_down(ajr,a,Gammap).dot(self.Q_power(m,ajr,Gamma))
        out=a*out[1,0]+(1-a)*out[0,1]
        #print('down {}:'.format(m),temp-out)
        return out

    def f(self,k,up=0,down=0,m=0,ajl=None,ajr=None):
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
            out=self.Q_power(k,aj=ajl).dot(self.Qp_up(aj=ajl))
            out = ajl*out[1,0]+(1-ajl)*out[0,1]

        elif up==down==1:
            out=self.Q_power(k,aj=ajl).dot(self.Qp_up(aj=ajl)).dot(self.Qp_down(aj=ajr)).dot(self.Q_power(m,aj=ajr))
            out = ajl*out[1,0]+(1-ajl)*out[0,1]

        elif up==0 and down==1:
            out=self.Qp_down(aj=ajr).dot(self.Q_power(m,aj=ajr))
            out = a*out[1,0]+(1-a)*out[0,1]

        elif down==0:
            out=self.Q_power(m,aj=ajr)
            out = ajr*out[1,0]+(1-ajr)*out[0,1]
        #
        # out=self.Q_power(k,aj=ajl).dot(
        #     np.linalg.matrix_power(self.Qp_up(aj=ajl),up).dot(
        #         np.linalg.matrix_power(self.Qp_down(aj=ajr),down).dot(
        #             self.Q_power(m,aj=ajr)
        #             )
        #         )
        #     )

        return out
        # if k+up>0:
        #     out = ajl*out[1,0]+(1-ajl)*out[0,1]
        # elif down>0:
        #     out = a*out[1,0]+(1-a)*out[0,1]
        # else:
        #     out = ajl*out[1,0]+(1-ajl)*out[0,1]
        return out
        # out=0.
        #
        #
        # try:
        #     if k+m+up+down==0:
        #         out=0.
        #     elif m+down==0:
        #         if up==0:
        #             out=self.f_left(k,**kwargs)
        #         else:
        #             out=self.f_up(k,**kwargs)
        #     elif k+up==0:
        #         if down==0:
        #             out=self.f_left(k,ajl=ajr,**kwargs)
        #         else:
        #             out=self.f_down(m,**kwargs)
        #     else:
        #         out=self.f_up_down(k,m,**kwargs)
        #
        # except ZeroDivisionError:
        #     pass
        # return out

    def weight_bias(self,f2,fk):
        out=0.
        #of E(w_l/w_r)=1+epsilon return epsilon
        if fk==0:
            if f2==0:
                out=0.
            else:
                out=f2*self.pattern_len
        else:
            out=-2*f2+f2*(self.pattern_len+2)*(1-(1-fk)**(self.pattern_len+1))/((self.pattern_len+1)*fk)
        return out

    def root_cluster_eq_ratio(self):
        #if ajl is None:
        ajl=self.high_child_prior
        #if ajr is None:
        ajr=self.high_child_prior
        #if a is None:
        a=self.root_prior
        #if Gamma is None:
        Gamma=self.flip_rate
        #if Gammap is None:
        Gammap=self.root_flip_rate
        kwargs={'c':self.c,'h':self.h,'L':self.pattern_len,'ajl':ajl,'ajr':ajr, 'a':a , 'Gamma':Gamma,'Gammap':Gammap}
        #eq prob of cluster divided by eq prob of articulation pt, here the root itself
        bias_list=[ self.weight_bias( self.f(0,1,1,0),self.f(self.h-1,0,0,0) ), self.weight_bias(self.f(0,0,1,1,ajl=a),self.f(self.h-1,1,0,0,ajr=a)) ]+[ self.weight_bias( self.f(0,0,0,2),self.f(self.h-1,1,1,m) )  for m in range(self.h-1) ]
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
        #if ajl is None:
        ajl=self.high_child_prior
        #if ajr is None:
        ajr=self.high_child_prior
        #if a is None:
        a=self.root_prior
        #if Gamma is None:
        Gamma=self.flip_rate
        #if Gammap is None:
        Gammap=self.root_flip_rate
        kwargs={'c':self.c,'h':self.h,'L':self.pattern_len,'ajl':ajl,'ajr':ajr, 'a':a , 'Gamma':Gamma,'Gammap':Gammap}

        e_r=self.weight_bias(self.f(2,0,0,0),self.f(self.h-2,0,0,0))
        e_u=self.weight_bias(self.f(1,1,0,0,ajr=a),self.f(self.h-2,0,0,0))

        bias_list=[ e_r,e_u ]+[ self.weight_bias( self.f(2,0,0,0),self.f(self.h-1+m,0,0,0) )  for m in range(self.h-2) ]
        out=1+(self.c-1)/( (self.c+bias_list[0])*(1+bias_list[1])+1+bias_list[0] )*\
            (
            np.sum([
                        self.c**(l-1)*(self.c+1+bias_list[l+1])/np.prod( [1+bias_list[k] for k in range(2,l+2)])
                for l in range(1,self.h-1) ])+\
            self.c**(self.h-2)/np.prod( [1+bias_list[k] for k in range(2,self.h)])
            )
        return out

    def eq_ratio(self,k):
        #eq prob of cluster divided by eq prob of articulation pt at height k over target
        #if ajl is None:
        ajl=self.high_child_prior
        #if ajr is None:
        ajr=self.high_child_prior
        #if a is None:
        a=self.root_prior
        #if Gamma is None:
        Gamma=self.flip_rate
        #if Gammap is None:
        Gammap=self.root_flip_rate
        kwargs={'c':self.c,'h':self.h,'L':self.pattern_len,'ajl':ajl,'ajr':ajr, 'a':a , 'Gamma':Gamma,'Gammap':Gammap}

        if k==0:
            return 1
        bias_list=[ self.weight_bias( self.f(2,0,0,0),self.f(l,0,0,0) )  for l in range(k-1,2*k-1) ]
        out=1+ (self.c-1)/(self.c+1+bias_list[0])*\
            (
            np.sum([
                        self.c**(l-1)*(self.c+1+bias_list[l])/np.prod( [ 1+bias_list[m] for m in range(1,l+1)])
                for l in range(1,k) ])+\
            self.c**(k-1)/np.prod( [ 1+bias_list[l] for l in range(1,k)])
            )
        return out

    def MF_mfpt(self,ajl=None,ajr=None,a=None,Gamma=None,Gammap=None,**kwargs):
        #if ajl is None:
        ajl=self.high_child_prior
        #if ajr is None:
        ajr=self.high_child_prior
        #if a is None:
        a=self.root_prior
        #if Gamma is None:
        Gamma=self.flip_rate
        #if Gammap is None:
        Gammap=self.root_flip_rate
        kwargs={'c':self.c,'h':self.h,'L':self.pattern_len,'ajl':ajl,'ajr':ajr, 'a':a , 'Gamma':Gamma,'Gammap':Gammap}
        #just under the root things are a bit messy, hence the following epsilons
        e_r=self.weight_bias(self.f(2,0,0,0),self.f(self.h-2,0,0,0))
        e_u=self.weight_bias(self.f(1,1,0,0,ajr=a),self.f(self.h-2,0,0,0))
        cord_weight_list = [ (1+self.weight_bias(self.f(2,0,0,0),self.f(k,0,0,0)))/(self.c+1+self.weight_bias(self.f(2,0,0,0),self.f(k,0,0,0))) for k in range(0,self.h-2) ] + [ (1+e_r)*(1+e_u)/( (self.c+e_r)*(1+e_u)+1+e_u ) ] + [ (1+self.weight_bias(self.f(0,1,1,0),self.f(self.h-1,0,0,0)))/(self.c+self.weight_bias(self.f(0,1,1,0),self.f(self.h-1,0,0,0))) ]
        eq_ratio_list = [ self.eq_ratio(k) for k in range(1,self.h-1) ]+[self.sub_root_cluster_eq_ratio()]+[ self.root_cluster_eq_ratio() ]
        out = np.sum(np.sum( [[eq_ratio_list[k]/cord_weight_list[k] for k in range(i,self.h)] for i in range(self.h)] ))
        return out

    def MTM(self, number_samples: int) ->np.array:
        #return the average transition matrix, sampled over number_samples interations.
        W=np.zeros( (len(self),len(self)) )
        node_order=[self.root]+list( self.nodes -set([self.root,self.target_node])  )+[self.target_node]
        for _ in range(number_samples):
            self.reset_patterns()
            W_temp=nx.to_numpy_array(self,nodelist=node_order)
            if (np.sum(W_temp,axis=-1)!=1).any:
                W_temp=np.diag(1/np.sum(W_temp,axis=-1)).dot(W_temp)
            W+=W_temp
        W/=number_samples
        return W

    def MTM_mfpt(self, number_samples: int) -> np.float():
        W=self.MTM(number_samples)
        out = np.sum( np.linalg.inv( np.eye(len(self)-1) - W[:-1,:-1] ),axis=-1 )[0]
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
        self.O_ll=np.sum(self.U_list)/(self.pattern_len*(self.c-1))#1+self.O_hh-2*(2*self.overlap/self.pattern_len+1/self.c)  #max(0,1-(2/self.c+4*self.overlap/self.pattern_len))
        self.O_hl=(1-self.O_hh-self.O_ll)/2
        self.O_lh=self.O_hl

    def f(self,k,up,down,m,mu=2,**kwargs):
        #if ajl is None:
        a_h=self.high_child_prior
        #if ajr is None:
        a_l=self.low_child_prior
        #if a is None:
        a=self.root_prior
        #if Gamma is None:
        Gamma=self.flip_rate
        #if Gammap is None:
        Gammap=self.root_flip_rate

        out=0.
        coordinates=(k,up,down,m)

        if up==1 and down==0:
            f_h=MF_patternWalker.f(self,*coordinates,ajl=a_h,ajr=a)
            f_l=MF_patternWalker.f(self,*coordinates,ajl=a_l,ajr=a)
            out=f_h*(self.pattern_len/self.c+2*self.overlap)/self.pattern_len+f_l*(self.pattern_len-self.pattern_len/self.c-2*self.overlap)/self.pattern_len

        elif up==0 and down==1:
            f_h=MF_patternWalker.f(self,*coordinates,ajl=a,ajr=a_h)
            f_l=MF_patternWalker.f(self,*coordinates,ajl=a,ajr=a_l)
            out=f_h*(self.pattern_len/self.c+2*self.overlap)/self.pattern_len+f_l*(self.pattern_len-self.pattern_len/self.c-2*self.overlap)/self.pattern_len

        elif up==0 and down+m==0:
            f_h=MF_patternWalker.f(self,*coordinates,ajl=a_h,ajr=a_h)
            f_l=MF_patternWalker.f(self,*coordinates,ajl=a_l,ajr=a_l)
            out=f_h*(self.pattern_len/self.c+2*self.overlap)/self.pattern_len+f_l*(self.pattern_len-self.pattern_len/self.c-2*self.overlap)/self.pattern_len

        elif up+k==0 and down==0:
            f_h=MF_patternWalker.f(self,*coordinates,ajl=a_h,ajr=a_h)
            f_l=MF_patternWalker.f(self,*coordinates,ajl=a_l,ajr=a_l)
            out=f_h*(self.pattern_len/self.c+2*self.overlap)/self.pattern_len+f_l*(self.pattern_len-self.pattern_len/self.c-2*self.overlap)/self.pattern_len

        else:
            f_hh=MF_patternWalker.f(self,*coordinates,ajl=a_h,ajr=a_h)
            f_hl=MF_patternWalker.f(self,*coordinates,ajl=a_h,ajr=a_l)
            f_lh=MF_patternWalker.f(self,*coordinates,ajl=a_l,ajr=a_h)
            f_ll=MF_patternWalker.f(self,*coordinates,ajl=a_l,ajr=a_l)
            if mu==0:
                out=self.O_hh*f_hh+self.O_lh*f_lh+self.O_hl*f_hl+self.O_ll*f_ll
            else:
                out=self.O_list[mu-2]/self.pattern_len*f_hh+(self.pattern_len-self.U_list[mu-2]-self.O_list[mu-2])/self.pattern_len*(f_hl+f_lh)/2+self.U_list[mu-2]/self.pattern_len*f_ll

        #
        # if down+m==0 or k+up==0:
        #
        #     f_h=MF_patternWalker.f(self,*coordinates,ajl=self.high_child_prior,ajr=self.high_child_prior)
        #     f_l=MF_patternWalker.f(self,*coordinates,ajl=self.low_child_prior,ajr=self.low_child_prior)
        #     out=f_h*(self.pattern_len/self.c+2*self.overlap)/self.pattern_len+f_l*(self.pattern_len-self.pattern_len/self.c-2*self.overlap)/self.pattern_len
        # # elif k+up==0:
        # #     f_h=MF_patternWalker.f(self,*coordinates,ajl=self.high_child_prior,ajr=self.high_child_prior)
        # #     f_l=MF_patternWalker.f(self,0,up=0,down=0,m=m,ajl=self.low_child_prior,ajr=self.low_child_prior)
        # #     out=f_h*(self.pattern_len/self.c+2*self.overlap)/self.pattern_len+f_l*(self.pattern_len-self.pattern_len/self.c-2*self.overlap)/self.pattern_len
        # else:
        #     f_hh=MF_patternWalker.f(self,*coordinates,ajl=self.high_child_prior,ajr=self.high_child_prior)
        #     f_hl=MF_patternWalker.f(self,*coordinates,ajl=self.high_child_prior,ajr=self.low_child_prior)
        #     f_lh=MF_patternWalker.f(self,*coordinates,ajl=self.low_child_prior,ajr=self.high_child_prior)
        #     f_ll=MF_patternWalker.f(self,*coordinates,ajl=self.low_child_prior,ajr=self.low_child_prior)
        #     out=self.O_hh*f_hh+self.O_hl*f_hl+self.O_lh*f_lh+self.O_ll*f_ll

        return out

    def f_left(self,k,**kwargs):
        f_h=MF_patternWalker.f(self,k,up=0,down=0,m=0,ajl=self.high_child_prior)
        f_l=MF_patternWalker.f(self,k,up=0,down=0,m=0,ajl=self.low_child_prior)
        print(f_h,f_l)
        out=f_h*(self.pattern_len/self.c+2*self.overlap)/self.pattern_len+f_l*(self.pattern_len-self.pattern_len/self.c-2*self.overlap)/self.pattern_len
        return out

    def f_up(self,k,**kwargs):
        f_h=MF_patternWalker.f(self,k,up=1,down=0,m=0,ajl=self.high_child_prior)
        f_l=MF_patternWalker.f(self,k,up=1,down=0,m=0,ajl=self.low_child_prior)
        out=f_h*(self.pattern_len/self.c+2*self.overlap)/self.pattern_len+f_l*(self.pattern_len-self.pattern_len/self.c-2*self.overlap)/self.pattern_len
        return out

    def f_up_down(self,k,m=1,**kwargs):
        f_hh=MF_patternWalker.f(self,k,up=1,down=1,m=m,ajl=self.high_child_prior,ajr=self.high_child_prior)
        f_hl=MF_patternWalker.f(self,k,up=1,down=1,m=m,ajl=self.high_child_prior,ajr=self.low_child_prior)
        f_lh=MF_patternWalker.f(self,k,up=1,down=1,m=m,ajl=self.low_child_prior,ajr=self.high_child_prior)
        f_ll=MF_patternWalker.f(self,k,up=1,down=1,m=m,ajl=self.low_child_prior,ajr=self.low_child_prior)
        #print(f_hh,f_hl,f_lh,f_ll)
        out=self.O_hh*f_hh+self.O_hl*f_hl+self.O_lh*f_lh+self.O_ll*f_ll
        #print(MF_patternWalker.f_up_down(self,k,m,ajl=self.high_child_prior,ajr=self.high_child_prior),out)
        return out

    def f_down(self,m,**kwargs):
        f_h=MF_patternWalker.f(self,0,up=0,down=0,m=m,ajr=self.high_child_prior)
        f_l=MF_patternWalker.f(self,0,up=0,down=0,m=m,ajr=self.low_child_prior)
        out=f_h*(self.pattern_len/self.c+2*self.overlap)/self.pattern_len+f_l*(self.pattern_len-self.pattern_len/self.c-2*self.overlap)/self.pattern_len
        return out

    def root_cluster_eq_ratio(self):
        #if ajl is None:
        ajl=self.high_child_prior
        #if ajr is None:
        ajr=self.high_child_prior
        #if a is None:
        a=self.root_prior
        #if Gamma is None:
        Gamma=self.flip_rate
        #if Gammap is None:
        Gammap=self.root_flip_rate
        kwargs={'c':self.c,'h':self.h,'L':self.pattern_len,'ajl':ajl,'ajr':ajr, 'a':a , 'Gamma':Gamma,'Gammap':Gammap}
        #eq prob of cluster divided by eq prob of articulation pt, here the root itself
        bias_dict={mu: [self.weight_bias( self.f(0,1,1,0,mu),self.f(self.h-1,0,0,0,mu) ), self.weight_bias(self.f(0,0,1,1,mu,ajl=a),self.f(self.h-1,1,0,0,mu,ajr=a)) ]+[ self.weight_bias( self.f(0,0,0,2,mu),self.f(self.h-1,1,1,m,mu) )  for m in range(self.h-1)] for mu in range(2,self.c+1)}

        out=1+np.sum([
                1/(self.c+bias_dict[mu][0])*\
                (
                np.sum([
                self.c**(l-1)*(self.c+1+bias_dict[mu][l])/np.prod([1+bias_dict[mu][k] for k in range(1,l+1)])
                for l in range(1,self.h)])+\
                self.c**(self.h-1)/np.prod([1+bias_dict[mu][k] for k in range(1,self.h)])
                )
            for mu in range(2,self.c+1) ])
        #
        # out=1+ (self.c-1)/(self.c+bias_list[0])*\
        #     (
        #     np.sum([
        #             self.c**(l-1)*(self.c+1+bias_list[l])/np.prod( [1+bias_list[k] for k in range(1,l+1)])
        #             for l in range(1,self.h) ])+\
        #     self.c**(self.h-1)/np.prod( [1+bias_list[k] for k in range(1,self.h)])
        #     )
        return out

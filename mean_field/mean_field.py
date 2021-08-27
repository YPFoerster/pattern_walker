"""
Class for walker with approximate average-patterns, in particular the
approximate MFPT from root to target.
"""

import pattern_walker as rw
import numpy as np 

__all__ = [
    'MF_patternWalker', 'overlap_MF_patternWalker'
    ]


class MF_patternWalker(rw.fullProbPatternWalker):
    #does all of the above with the correct parameters as given by G
    def __init__(self,c=4,h=3,*args,**params):
        self.c=c
        self.h=h
        super(MF_patternWalker,self).__init__(*args,**params)

    def Q_power(self,k,ajl=None,Gamma=None):
        if ajl is None:
            ajl=self.high_child_prior
        if Gamma is None:
            Gamma=self.flip_rate

        if ajl>0:
#             out= np.array(
#                     [[1-ajl*(1-(1-Gamma/ajl)**k),ajl*(1-(1-Gamma/ajl)**k)],
#                              [(1-ajl)*(1-(1-Gamma/ajl)**k),1-(1-ajl)*(1-(1-Gamma/ajl)**k)]]
#                     )
            out= np.array(
                    [[1-ajl*(1-(1-Gamma)**k),ajl*(1-(1-Gamma)**k)],
                             [(1-ajl)*(1-(1-Gamma)**k),1-(1-ajl)*(1-(1-Gamma)**k)]]
                    )
        elif ajl==0:
#             out=np.linalg.matrix_power(np.array( [[1-Gamma,Gamma],[Gamma,1-Gamma]] ),k)
            out=np.linalg.matrix_power(np.array( [[1,0],[0,1]] ),k)
        return out

    def Qp_up(self,ajl=None,a=None,Gammap=None):
        if ajl is None:
            ajl=self.high_child_prior
        if a is None:
            a=self.root_prior
        if Gammap is None:
            Gammap=self.root_flip_rate

        if ajl>0:
            out= np.array(
                    [[(1-a)*(1-Gammap)/(1-ajl),(a-ajl+(1-a)*Gammap)/(1-ajl)],
                     [(1-a)/ajl*Gammap,1-(1-a)/ajl*Gammap]]
                    )
        elif ajl==0:
            out=np.array(
                    [[1-Gammap,Gammap],
                     [Gammap,1-Gammap]]
                    )
        return out

    def Qp_down(self,ajr=None,a=None,Gammap=None):
        if ajr is None:
            ajr=self.high_child_prior
        if a is None:
            a=self.root_prior
        if Gammap is None:
            Gammap=self.root_flip_rate

        if a>0:
            return np.array(
                    [[1-Gammap,Gammap],
                     [1-(ajr-(1-a)*Gammap)/a,(ajr-(1-a)*Gammap)/a]]
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
        out=(1-(1-Gamma/ajl)**k)
        return out

    def f_left(self,k,ajl=None,ajr=None,a=None,Gamma=None,Gammap=None,**kwargs):
        if ajl is None:
            ajl=self.high_child_prior
        if Gamma is None:
            Gamma=self.flip_rate

        #no root involved
        #checked and found to be correct
        temp=2*ajl*(1-ajl)*(1-(1-Gamma/ajl)**k)
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
        temp= 2*(1-a)*Gammap+a-ajl+2*(1-(1-Gamma/ajl)**(k-1))*(1-a)*ajl*(1-Gammap/ajl)
        out=self.Q_power(k-1,ajl,Gamma).dot(self.Qp_up(ajl,a,Gammap))
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
            2*self.kappa(k-1,ajl)*(1-a)*(ajl-Gammap)*(ajr-Gammap)/a+\
            2*self.kappa(m-1,ajr)*(1-self.kappa(k-1,ajl))*(1-a)*(ajl-Gammap)*(ajr-Gammap)/a
        #last two edges go over the root, up and then down from leftmost to some branch on the right
        out=self.Q_power(k-1,ajl,Gamma).dot(self.Qp_up(ajl,a,Gammap).dot(self.Qp_down(ajr,a,Gammap).dot(self.Q_power(m-1,ajr,Gamma))))
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
        temp= 2*(1-a)*Gammap+a-ajr+2*self.kappa(m-1,ajr,Gamma)*(1-a)*(ajr-Gammap)
        out=self.Qp_down(ajr,a,Gammap).dot(self.Q_power(m-1,ajr,Gamma))
        out=a*out[1,0]+(1-a)*out[0,1]
        #print('down {}:'.format(m),temp-out)
        return out

    def f(self,k,up=0,down=0,m=0,ajl=None,ajr=None,a=None,Gamma=None,Gammap=None,**kwargs):
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

        out=0.
        try:
            if k==0 and m==0:
                out=0
            elif m==0:
                if up==0:
                    out=self.f_left(k,**kwargs)
                else:
                    out=self.f_up(k,**kwargs)
            elif k==0:
                if down==0:
                    out=self.f_left(k,ajl=ajr,**kwargs)
                else:
                    out=self.f_down(m,**kwargs)
            else:
                out=self.f_up_down(k,m,**kwargs)

        except ZeroDivisionError:
            pass
        return out

    def weight_bias(self,f2,fk):
        out=1.
        #return E(w_l/w_r)=1+epsilon
        if fk==0:
            if f2==0:
                out=1.
            else:
                out=1+f2*L
        else:
            out=1-2*f2+f2*(L+2)*(1-(1-fk)**(L+1))/((L+1)*fk)
        return out

    def root_cluster_eq_ratio(self,ajl=None,ajr=None,a=None,Gamma=None,Gammap=None,**kwargs):
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
        kwargs={'c':self.c,'h':self.h,'L':self.pattern_len,'ajl':ajl,'ajr':ajr, 'a':a , 'Gamma':Gamma,'Gammap':Gammap}
        #eq prob of cluster divided by eq prob of articulation pt, here the root itself
        bias_list=[ self.weight_bias( self.f(1,1,1,1,**kwargs),self.f(self.h-1,0,0,0,**kwargs) ), self.weight_bias(self.f(0,0,1,2,**kwargs),self.f(self.h,1,0,0,**kwargs)) ]+[ self.weight_bias( self.f(0,0,0,2,**kwargs),self.f(self.h,1,1,m,**kwargs) )  for m in range(1,self.h) ]
        out=1+np.sum([
                    (self.c-1)*self.c**(l-1)*(self.c+bias_list[l])/(self.c-1+bias_list[0])*np.prod( [1/(bias_list[k]) for k in range(1,l+1)])
            for l in range(1,self.h) ])+\
        (self.c-1)*self.c**(self.h-1)/(self.c-1+bias_list[0])*np.prod( [1/(bias_list[k]) for k in range(1,self.h)])
        return out

    def sub_root_cluster_eq_ratio(self,ajl=None,ajr=None,a=None,Gamma=None,Gammap=None,**kwargs):
        #eq prob of cluster divided by eq prob of articulation pt, here the node under the root
        #just under the root things are a bit messy, hence the following epsilons
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
        kwargs={'c':self.c,'h':self.h,'L':self.pattern_len,'ajl':ajl,'ajr':ajr, 'a':a , 'Gamma':Gamma,'Gammap':Gammap}

        e_r=self.weight_bias(self.f(2,0,0,0,**kwargs),self.f(self.h-2,0,0,0,**kwargs))
        e_u=self.weight_bias(self.f(2,1,0,0,**kwargs),self.f(self.h-2,0,0,0,**kwargs))

        bias_list=[ e_r*e_u ]+[ self.weight_bias( self.f(2,0,0,0,**kwargs),self.f(self.h-2+m,0,0,0,**kwargs) )  for m in range(1,self.h-1) ]
        out=1+np.sum([
                    (self.c-1)*self.c**(l-1)*(self.c+bias_list[l])/((self.c-1)*e_u+e_r+e_u*e_r)*np.prod( [1/(bias_list[k]) for k in range(1,l+1)])
            for l in range(1,self.h-1) ])+\
        (self.c-1)*self.c**(self.h-2)/((self.c-1)*e_u+e_r+e_u*e_r)*np.prod( [1/(bias_list[k]) for k in range(1,self.h-1)])
        return out

    def eq_ratio(self,k,ajl=None,ajr=None,a=None,Gamma=None,Gammap=None,**kwargs):
        #eq prob of cluster divided by eq prob of articulation pt at height k over target
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
        kwargs={'c':self.c,'h':self.h,'L':self.pattern_len,'ajl':ajl,'ajr':ajr, 'a':a , 'Gamma':Gamma,'Gammap':Gammap}

        if k==0:
            return 1
        bias_list=[ self.weight_bias( self.f(2,0,0,0,**kwargs),self.f(l,0,0,0,**kwargs) )  for l in range(k-1,2*k-1) ]
        out=1+np.sum([
                    (self.c-1)*self.c**(l-1)*(self.c+bias_list[l])/(self.c+bias_list[0])*np.prod( [1/(bias_list[m]) for m in range(1,l+1)])
            for l in range(1,k) ])+\
        (self.c-1)*self.c**(k-1)/(self.c+bias_list[0])*np.prod( [1/(bias_list[l]) for l in range(1,k)])
        return out

    def MF_mfpt(self,ajl=None,ajr=None,a=None,Gamma=None,Gammap=None,**kwargs):
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
        kwargs={'c':self.c,'h':self.h,'L':self.pattern_len,'ajl':ajl,'ajr':ajr, 'a':a , 'Gamma':Gamma,'Gammap':Gammap}
        #just under the root things are a bit messy, hence the following epsilons
        e_r=self.weight_bias(self.f(2,0,0,0,**kwargs),self.f(self.h-2,0,0,0,**kwargs))
        e_u=self.weight_bias(self.f(2,1,0,0,**kwargs),self.f(self.h-2,0,0,0,**kwargs))
        cord_weight_list = [ self.weight_bias(self.f(2,0,0,0,**kwargs),self.f(k,0,0,0,**kwargs))/(self.c+self.weight_bias(self.f(2,0,0,0,**kwargs),self.f(k,0,0,0,**kwargs))) for k in range(0,self.h-2) ] + [ e_r*e_u/((self.c-1)*e_u+e_r+e_u*e_r) ] + [ self.weight_bias(self.f(1,1,1,1,**kwargs),self.f(self.h-1,0,0,0,**kwargs))/(self.c-1+self.weight_bias(self.f(1,1,1,1,**kwargs),self.f(self.h-1,0,0,0,**kwargs))) ]
        eq_ratio_list = [ self.eq_ratio(k+1) for k in range(0,self.h-2) ]+[self.sub_root_cluster_eq_ratio()]+[ self.root_cluster_eq_ratio() ]
        out = np.sum(np.sum( [[eq_ratio_list[self.h-k-1]/cord_weight_list[self.h-k-1] for k in range(i)] for i in range(1,self.h+1)] ))
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
        self.O_hh=np.sum(self.O_list)/(self.pattern_len*(self.c-1))
        self.O_ll=1+self.O_hh-2*(2*self.overlap/self.pattern_len+1/self.c)  #max(0,1-(2/self.c+4*self.overlap/self.pattern_len))
        self.O_hl=(1-self.O_hh-self.O_ll)/2
        self.O_lh=self.O_hl

    def f_left(self,k,**kwargs):
        f_h=MF_patternWalker.f_left(self,k,ajl=self.high_child_prior)
        f_l=MF_patternWalker.f_left(self,k,ajl=self.low_child_prior)
        out=f_h*(self.pattern_len/self.c+2*self.overlap)/self.pattern_len+f_l*(self.pattern_len-self.pattern_len/self.c-2*self.overlap)/self.pattern_len
        return out

    def f_up(self,k,**kwargs):
        f_h=MF_patternWalker.f_up(self,k,ajl=self.high_child_prior)
        f_l=MF_patternWalker.f_up(self,k,ajl=self.low_child_prior)
        out=f_h*(self.pattern_len/self.c+2*self.overlap)/self.pattern_len+f_l*(self.pattern_len-self.pattern_len/self.c-2*self.overlap)/self.pattern_len
        return out

    def f_up_down(self,k,m=1,**kwargs):
        f_hh=MF_patternWalker.f_up_down(self,k,m,ajl=self.high_child_prior,ajr=self.high_child_prior)
        f_hl=MF_patternWalker.f_up_down(self,k,m,ajl=self.high_child_prior,ajr=self.low_child_prior)
        f_lh=MF_patternWalker.f_up_down(self,k,m,ajl=self.low_child_prior,ajr=self.high_child_prior)
        f_ll=MF_patternWalker.f_up_down(self,k,m,ajl=self.low_child_prior,ajr=self.low_child_prior)
        #print(f_hh,f_hl,f_lh,f_ll)
        out=self.O_hh*f_hh+self.O_hl*f_hl+self.O_lh*f_lh+self.O_ll*f_ll
        #print(MF_patternWalker.f_up_down(self,k,m,ajl=self.high_child_prior,ajr=self.high_child_prior),out)
        return out

    def f_down(self,m,**kwargs):
        f_h=MF_patternWalker.f_down(self,m,ajr=self.high_child_prior)
        f_l=MF_patternWalker.f_down(self,m,ajr=self.low_child_prior)
        out=f_h*(self.pattern_len/self.c+2*self.overlap)/self.pattern_len+f_l*(self.pattern_len-self.pattern_len/self.c-2*self.overlap)/self.pattern_len
        return out

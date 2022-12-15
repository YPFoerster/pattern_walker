from pattern_walker import fullProbPatternWalker as pw
from pattern_walker.utils.plotting import plot_tree
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider,Button
from networkx.drawing.nx_agraph import graphviz_layout
import networkx as nx

class walkerAnimation(pw):
    def __init__(self,c,h,*args,**kwargs):
        pw.__init__(self,*args,**kwargs)
        self.set_weights()
        self.h=h #height of the tree
        self.c=c #coordinatio number
        self.tau = np.power(1-self.Gamma,self.h-1)

        self.fig, self.ax = plt.subplots(1,1)

        self.pos=graphviz_layout(self.hierarchy_backup,prog='dot') #Positions can be taken from H
        nx.draw_networkx_nodes(self, self.pos, node_color='k',node_size=100,ax=self.ax)
        nx.draw_networkx_nodes(self, self.pos,nodelist=[self.target_node],node_color='y',node_size=100,ax=self.ax)

        self.edges_collection = self.draw_edges()

        self.current_x_plot = nx.draw_networkx_nodes(self,self.pos,
                                                     nodelist=[self.x],node_color='r',
                                                     node_size=100, ax=self.ax
                                                    )

        self.fig.subplots_adjust( bottom=0.5)

        self.s_tau = paramField(self.fig,[0.1, 0.05, 0.35, 0.03],r'$\tau$', 0., 1., valinit=self.tau, valstep=0.1 )
        self.s_Gamma_root = paramField(self.fig,[0.1, 0.1, 0.35, 0.03],r'$\Gamma^\prime$', 0., 1., valinit=self.Gamma_root, valstep=0.1 )

        self.s_a_high =  paramField(self.fig,[0.1, 0.15, 0.35, 0.03],r'$a_h$', 0., 1., valinit=self.a_high, valstep=0.1 )
        self.s_a_low = paramField(self.fig,[0.1, 0.2, 0.35, 0.03],r'$a_l$', 0., 1., valinit=self.a_low, valstep=0.1 )
        self.s_beta_low = paramField(self.fig,[0.1, 0.25, 0.35, 0.03],r'$\beta_l$', 0., 1., valinit=0., valstep=0.1 )

        self.s_overlap = paramField(self.fig,[0.1, 0.3, 0.35, 0.03],r'$\Delta$', 0., int(self.pattern_len*(self.c-2)/2), valinit=self.overlap, valstep=1 )
#         self.ax_a = plt.axes([0.1, 0.1, 0.35, 0.03])#, facecolor=axcolor)
#         s_a = Slider(ax_a, r'$a$', 0., 1., valinit=a_root, valstep=0.1)

        self.ax_reset_walker = self.fig.add_axes([0.7, 0.05, 0.2, 0.075])
        self.b_reset_walker = Button(self.ax_reset_walker, 'Reset Walker')
        self.b_reset_walker.on_clicked(self.reset_x)


        self.ax_set = self.fig.add_axes([0.7, 0.15, 0.2, 0.075])
        self.b_set = Button(self.ax_set, 'New Patterns')
        self.b_set.on_clicked(self.update_params)

    def get_edge_weights_iter(self):
        return zip(*nx.get_edge_attributes(self,'weight').items())

    def draw_edges(self):
        (edges,weights) = self.get_edge_weights_iter()
        return nx.draw_networkx_edges(self,
                                self.pos, edgelist=edges,
                                edge_color=weights,
                                edge_cmap=plt.cm.Blues,ax=self.ax
                                                    )


    def set_Gamma(self,tau=None,Gamma=None):
        if Gamma is None:
            self.Gamma = 1-np.power(tau,1/(self.h-1))
            self.tau = tau
        else:
            self.tau = np.power(1-Gamma,self.h-1)
            self.Gamma = Gamma


    def update_params(self,t):
        self.a_high = self.s_a_high.val
        self.Gamma_root = self.s_Gamma_root.val
        self.a_low=self.s_a_low.val
        self.a_root = self.a_high-self.a_low+self.s_beta_low.val
        self.overlap=self.s_overlap.val

        tau=self.s_tau.val
        self.set_Gamma(tau=tau)
        self.reset_patterns()
        self.edges_collection = []
        self.edges_collection = self.draw_edges()

    def reset_x(self,t):
        self.reset_walker()

    def update(self,frame):
#         global current_plot,G
        self.current_x_plot.remove()
        self.current_x_plot = nx.draw_networkx_nodes(self,self.pos,
                                                   nodelist=[self.x],
                                                   node_color='r',
                                                   node_size=100, ax=self.ax
                                                  )



        if self.x!=self.target_node:
            self.step()
        else:
            self.reset_walker()



        return self.current_x_plot,




    def anim(self):
        self.fig.canvas.draw_idle()
        return FuncAnimation(self.fig, self.update,blit=True, frames=200,  interval=1000)


class paramField(Slider):
    def __init__(self,fig,coords,*args,**kwargs):
        self.ax = fig.add_axes(coords)
#         self.ax=ax
        Slider.__init__(self,self.ax, *args,**kwargs)

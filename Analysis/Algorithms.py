#implementation of algorithms
import math
import numpy as np
import time as time
from cmath import sqrt as isqrt

from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.gridspec as gridspec
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.animation as animation

import numpy.matlib
import numpy.polynomial as npoly
import random

from scipy import stats
from scipy.spatial import ConvexHull
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import cdist, jensenshannon
from scipy.optimize import linear_sum_assignment, curve_fit
from scipy.stats import rankdata, ortho_group, percentileofscore, spearmanr

from sklearn.metrics import pairwise_distances, r2_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.decomposition import PCA

import persim
import persim.landscapes
from ripser import ripser as tda
from persim import plot_diagrams
import umap.umap_ as umap

class Manifold_Generator:
    def __init__(self):
        return None

    def __call__(self, amount_of_points, manifold_type, *args):
        return getattr(self, manifold_type)(amount_of_points, args)
    
    def R3(self, amount_of_points, *args):
        x,y,z = np.meshgrid(np.linspace(-args[0],args[0],amount_of_points),
                        np.linspace(-args[1],args[1],amount_of_points),
                        np.linspace(-args[2],args[2],amount_of_points))
        R = np.vstack([x.flatten(),y.flatten(),z.flatten()])
        return R
    
    def S1(self, amount_of_points, *args):
        r, theta = args[0], np.linspace(0, 2*np.pi, amount_of_points)
        return np.array([r*np.cos(theta), r*np.sin(theta)])

    def S2(self, amount_of_points, *args):
        r, phi, theta  = args[0], np.linspace(0, np.pi, amount_of_points), np.linspace(0, 2*np.pi, amount_of_points)
        Phi, Theta = np.meshgrid(phi, theta)
        Phi, Theta = Phi.flatten(), Theta.flatten()
        return np.array([r*np.cos(Theta)*np.sin(Phi), r*np.sin(Theta)*np.sin(Phi), r*np.cos(Phi)])

    def T2(self, amount_of_points, *args):
        R, r, phi, theta  = args[0], args[1], np.linspace(0, 2*np.pi, amount_of_points,endpoint=False) ,np.linspace(0, 2*np.pi, amount_of_points,endpoint=False)
        Phi, Theta = np.meshgrid(phi,theta)
        Phi, Theta = Phi.flatten(), Theta.flatten()
        return np.array([(R+r*np.cos(Theta))*np.cos(Phi), (R+r*np.cos(Theta))*np.sin(Phi), r*np.sin(Theta)]) 
    
    def Sn(self, amount_of_points, *args):
        x = np.random.randn(amount_of_points**2,args[0])
        x = x.T/np.linalg.norm(x,axis=1)
        return x

    def KB(self,amount_of_points,*args):
        phi = np.linspace(0,np.pi,amount_of_points)
        theta = np.linspace(0,2*np.pi,amount_of_points)
        Phi, Theta = np.meshgrid(phi,theta)
        Phi_KB, Theta_KB = Phi.flatten(), Theta.flatten()
        x = (-2/15)*np.cos(Phi_KB)*(3*np.cos(Theta_KB)-30*np.sin(Phi_KB)+90*np.sin(Phi_KB)*np.cos(Phi_KB)**2
                           -60*np.sin(Phi_KB)*np.cos(Phi_KB)**6+5*np.cos(Phi_KB)*np.cos(Theta_KB)*np.sin(Phi_KB))
        y = (-1/15)*np.sin(Phi_KB)*(3*np.cos(Theta_KB)-3*np.cos(Theta_KB)*np.cos(Phi_KB)**2-48*np.cos(Theta_KB)*np.cos(Phi_KB)**2
                                    +48*np.cos(Theta_KB)*np.cos(Phi_KB)**6-60*np.sin(Phi_KB)+5*np.cos(Phi_KB)*np.cos(Theta_KB)*np.sin(Phi_KB)
                                    -5*np.cos(Theta_KB)*np.sin(Phi_KB)*np.cos(Phi_KB)**3-80*np.cos(Theta_KB)*np.sin(Phi_KB)*np.cos(Phi_KB)**5
                                    +80*np.cos(Theta_KB)*np.sin(Phi_KB)*np.cos(Phi_KB)**7)
        z = (2/15)*(3+5*np.cos(Phi_KB)*np.sin(Phi_KB))*np.sin(Theta_KB)
        KB = np.array([x,y,z])
        return KB 
        
class Persistent_Homology:
    def __init__(self):
        None

    def __call__(self, manifold, metric, normalized, *args):
        distance_matrix, birth_death_diagram = self.homology_analysis(manifold, metric, *args)
        return [distance_matrix, self.normalize(birth_death_diagram) if normalized else birth_death_diagram]

    def homology_analysis(self, manifold, metric, *args):
        '''
            Computes persistent homology
            -----------------------
            Outputs the distance_matrix and birth_death_diagram for the given manifold and metric. 
        '''
        distance_matrix = metric(manifold)
        birth_death_diagram = tda(distance_matrix, distance_matrix=True , maxdim=args[0], n_perm=args[1])['dgms']   
        return distance_matrix, birth_death_diagram

    def normalize(self, birth_death_diagram):
        '''
            Normalized birth/death distance
            -----------------------
            Outputs the persistence of a feature normalized between 0 and 1. 
        '''
        birth_death_diagram_copy = np.copy(birth_death_diagram)
        a = np.concatenate(birth_death_diagram_copy).flatten()
        finite_dgm = a[np.isfinite(a)]
        ax_min, ax_max = np.min(finite_dgm), np.max(finite_dgm)
        x_r = ax_max - ax_min

        buffer = x_r / 5

        x_down = ax_min - buffer / 2
        x_up = ax_max + buffer

        y_down, y_up = x_down, x_up
        yr = y_up - y_down
        b_inf = y_down + yr * 0.95
        norm_pers = []
        for i in range(len(birth_death_diagram)):
            norm_pers.append((birth_death_diagram[i])/b_inf)
        return norm_pers    

class Betti:
    def __init__(self):
        None

    def histograms(self, birth_death_diagram, binslst=[20,15,10]):
        ''' 
            Betti number persistance histograms
            -----------------------
            Outputs a list of histograms for persistance/birth/death for all specified Betti numbers.
        '''
        histograms = []
        persistant_values = []
        for i in range(len(birth_death_diagram)):
            persistant_values.append(birth_death_diagram[i][:,1] - birth_death_diagram[i][:,0])
            b_i = np.histogram(birth_death_diagram[i][:,1] - birth_death_diagram[i][:,0], np.linspace(0, 0.1, binslst[i]), density=True)
            histograms.append(b_i)
        return [persistant_values, histograms]
        
    def curves(self, birth_death_diagram, number_of_curves=3, epsilon = 0.001, duration=1000):
        '''
            Betti curves
            -----------------------
            Outputs a list of Betti curves, indicating how many n-holes there are for a given epsilon value.
        '''
        curves = []
        finite_dgm = birth_death_diagram[0][np.isfinite(birth_death_diagram[0])]
        x_maxs = []
        for i in range(len(birth_death_diagram)):
            if len(birth_death_diagram[i])>0 and i!=0:
                x_maxs.append(np.max(birth_death_diagram[i]))
            else:
                x_maxs.append(np.max(finite_dgm))
        e = np.copy(epsilon)
        for i in range(len(birth_death_diagram)):
            Bn = np.zeros(duration)
            for j in range(duration):
                count = 0 
                for k in range(len(birth_death_diagram[i][:,0])):
                    if birth_death_diagram[i][k,0]<epsilon and birth_death_diagram[i][k,1]>epsilon:
                        count = count + 1
                Bn[j] = count
                epsilon = epsilon + e
            epsilon = np.copy(e)
            curves.append(Bn)
        return [np.linspace(0, epsilon*duration, duration), curves]

# class Eigendiffusion:
#     def __init__(self,):
        
        
        
        
    
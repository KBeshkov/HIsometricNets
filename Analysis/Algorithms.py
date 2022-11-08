#implementation of algorithms
import math
import numpy as np

from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.gridspec as gridspec
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.animation as animation

import numpy.matlib
import numpy.polynomial as npoly

from scipy import stats
from scipy.spatial import ConvexHull
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import cdist, jensenshannon
from scipy.optimize import linear_sum_assignment, curve_fit
from scipy.stats import rankdata, ortho_group, percentileofscore, spearmanr

from sklearn.metrics import pairwise_distances, r2_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.decomposition import PCA

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

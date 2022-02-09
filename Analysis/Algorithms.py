#implementation of algorithms
import numpy as np
import math
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d import Axes3D 
import numpy.matlib
from scipy import stats
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import cdist, jensenshannon
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata
from sklearn.metrics import pairwise_distances, r2_score
from sklearn.linear_model import Lasso, Ridge
from ripser import ripser as tda
from persim import plot_diagrams
# import umap.umap_ as umap
from sklearn.decomposition import PCA
# import hdbscan
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.gridspec as gridspec
import time as time
from scipy.stats import ortho_group, percentileofscore
from cmath import sqrt as isqrt
from scipy.spatial import ConvexHull
import persim

    
'''
    Normalized birth/death distance:
    Outputs the persistence of a feature normalized between 0 and 1. In other words
    (death_of_feature-birth_of_feature)/upper_bound

    Input: a birth/death diagram
    
    Output: a birth/death diagram with normalized distances
'''

def normal_bd_dist(x):
    x_copy = np.copy(x)
    a = np.concatenate(x_copy).flatten()
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
    for i in range(len(x)):
        norm_pers.append((x[i])/b_inf)
    return norm_pers    
 
''' 
    Betti number persistance histograms
    -----------------------
    Outputs histograms of betti number persistance/birth/death for all specified betti numbers
    
    Input: a birth/death diagram
    bins: the number of bins in each histogram
    plots: if true creates a plot of the histograms
    
    Output: a list of histograms for each betti number diemension
'''

def Betti_hist(B,binslst=[20,15,10],plots=False):
    Bhist = []
    pers_vals = []
    for i in range(len(B)):
        pers_vals.append(B[i][:,1]-B[i][:,0])
        b_i = np.histogram(B[i][:,1]-B[i][:,0],np.linspace(0,0.1,binslst[i]),density=True)
        Bhist.append(b_i)
    if plots==True:
        plt.figure()
        for i in range(len(B)):
            plt.plot(Bhist[i][1][:-1],Bhist[i][0])
    return [pers_vals,Bhist]
       

 
'''
    Betti curves
    ----------------------
    Outputs the betti curves, indicating how many n-holes there are for a given
    epsilon value
    
    Input: a birth/death diagram
    
    Output: a list of curves
'''

def Betti_curves(B,N_curves = 3,epsilon = 0.001,duration=1000):
    Bcurves = []
    finite_dgm = B[0][np.isfinite(B[0])]
    x_maxs = []
    for i in range(len(B)):
        if len(B[i])>0 and i!=0:
            x_maxs.append(np.max(B[i]))
        else:
            x_maxs.append(np.max(finite_dgm))
    e = np.copy(epsilon)
    for i in range(len(B)):
        Bn = np.zeros(duration)
        for j in range(duration):
            count = 0 
            for k in range(len(B[i][:,0])):
                if B[i][k,0]<epsilon and B[i][k,1]>epsilon:
                    count = count + 1
            Bn[j] = count
            epsilon = epsilon + e
        epsilon = np.copy(e)
        Bcurves.append(Bn)
    return [np.linspace(0,epsilon*duration,duration),Bcurves]


'''Numeric calcultion of geodesics using a shortest path graph algorithm'''
def geodesic(X,r=-1,eps=0.1,count=1):
    Xn = np.copy(X)
    if r>0:
        N = len(Xn)
        d = pairwise_distances(Xn)
        d_geod = (10**10)*np.ones([N,N])
        neighbors = []
        for i in range(N):
            neighbors=np.where(d[i,:]<r)[0]
            d_geod[i,neighbors] = d[i,neighbors]
        d_geod = shortest_path(d_geod)
        if np.sum(d_geod>=10**10)>0:
            count += 1
            dn = geodesic(Xn,r=r+eps,eps=eps,count=count)
            return dn
        else:
#            print('finished in ' + str(count) + ' recursions')
            return d_geod
    else:
        N = len(Xn)
        d = pairwise_distances(Xn)
        hom = tda(d,distance_matrix=True,maxdim=0)['dgms'][0]
        r = hom[-2,1]+eps*hom[-2,1]
        d_geod = (10**10)*np.ones([N,N])
        neighbors = []
        for i in range(N):
            neighbors=np.where(d[i,:]<r)[0]
            d_geod[i,neighbors] = d[i,neighbors]
        d_geod = shortest_path(d_geod)
        if np.sum(d_geod>=10**10)>0:
            count += 1
            dn = geodesic(Xn,r=r+0.1*r,eps=0.1*r,count=count)
            return dn
#        else:
#            print('finished in ' + str(count) + ' recursions')

        return d_geod



def extract_cycles(dmat,cocycle,threshold,all_generators = False):
    S = []
    strt = cocycle[0]
    fin = cocycle[1]
    adj_mat = np.copy(dmat)
    adj_mat[dmat>=threshold] = 0
    adj_mat[dmat<=threshold] = 1
    adj_mat[strt,fin] = 0
    adj_mat[fin,strt] = 0
    a = shortest_path(adj_mat,directed=False,return_predecessors=True,unweighted=True,indices=[strt,fin])[1]
    c = a[0,fin]  
    S.append(fin)
    S.append(c)
    while c != strt:
        S.append(a[0,c])
        c = a[0,c]
    if all_generators == True:
        S.append(np.unique(np.where(dmat[S,:]<=threshold)[1]))
        return(np.unique(S[-1]))
    else:
        return S
    
    
''' Earth mover's distance as a linear assignment problem:
    
    Input:
        X: an array of inputs (eg spikes)
        Y: an array of inputs
        L1 = a list of cells that are being used in X
        L2 = a list of cells that are being used in Y
        D: a precomputed disance matrix
        
    Output: the distance between clusters
'''    

    
def EM_dist(X,Y,L1 = 0,L2 = 0,D = 0,norm='euclidean'):
    if norm=='euclidean':
        d = cdist(X,Y)
        em_dist = np.sum(d[linear_sum_assignment(d)])
        return em_dist
    elif norm=='spiking':
        L = np.ix_(L1,L2)
        d = D[L]
        em_dist = np.mean(d[linear_sum_assignment(d)])
        return em_dist
     
    
def order_complex(D):
    N = len(D[:,0])
    ord_mat = np.triu(D)
    np.fill_diagonal(ord_mat,0)
    Ord = rankdata(ord_mat.flatten(),method='dense').reshape(np.shape(D))
#    inv_ranks = np.sum(Ord==1)
    Ord = np.triu(Ord)+np.triu(Ord).T
    Ord = Ord #- inv_ranks
    np.fill_diagonal(Ord,0)
    return Ord/np.max(Ord)


#From: https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
def fibonacci_sphere(nsamples=1):
    indices = np.arange(0, nsamples, dtype=float) + 0.5
    phi = np.arccos(1 - 2*indices/nsamples)
    theta = np.pi*(1 + 5**0.5)*indices
    x, y, z = np.cos(theta)*np.sin(phi), np.sin(theta)*np.sin(phi), np.cos(phi)
    coords = np.vstack((x,y,z))
    return [coords,phi,theta]

def sample_nsphere(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec

def fnct(x):
    return x + 0.5*np.sin(2*x)

def f_inv(x,er = 10**-8):
    xn = x + 0.0001*np.random.randn()
    y = fnct(xn)
    while abs(y-x)>er:
        xn -= (y-x)/(1+np.cos(2*xn))
        y = fnct(xn)
    return xn

def fibonacci_3sphere(nsamples=1):
    indices = np.arange(0, nsamples)    
    psi = np.zeros(nsamples)
    for i in range(nsamples):        
        psi[i] = f_inv(indices[i]*np.pi/(nsamples+1))
        print(i)
    theta = np.arccos(1-2*(indices*(2**0.5)-np.floor(indices*(2**0.5))))
    phi = 2*np.pi*(indices*(3**0.5)-np.floor(indices*(3**0.5)))
    
    t, x, y, z = np.cos(psi), np.cos(theta)*np.sin(psi), np.sin(psi)*np.sin(theta)*np.cos(phi), np.sin(psi)*np.sin(theta)*np.sin(phi)
    coords = np.vstack((t,x,y,z))
    return coords
    
    
def barcode_plot(diagram,dims=2,norm_ax = 0):
    results = {}
    if norm_ax==0:
        largest_pers = 0
        for d in range(dims):
            results['h'+str(d)] = diagram[d]
            if np.max(diagram[d][np.isfinite(diagram[d])])>largest_pers:
                largest_pers = np.max(diagram[d][np.isfinite(diagram[d])])
    elif norm_ax!=0:
        largest_pers=norm_ax
    clrs = ['tab:blue','tab:orange','tab:green']#['b','r','g','m','c']
    diagram[0][~np.isfinite(diagram[0])] = largest_pers+0.1*largest_pers
    plot_prcnt = 0*np.ones(dims)
    to_plot = []
    for curr_h, cutoff in zip(diagram, plot_prcnt):
         bar_lens = curr_h[:,1] - curr_h[:,0]
         plot_h = curr_h[bar_lens >= np.percentile(bar_lens, cutoff)]
         to_plot.append(plot_h)
    fig = plt.figure(dpi=300,figsize=(14, 6))
    gs = gridspec.GridSpec(dims, 4)
    for curr_betti, curr_bar in enumerate(to_plot):
        ax = fig.add_subplot(gs[curr_betti, :])
        for i, interval in enumerate(reversed(curr_bar)):
            plt.plot([interval[0], interval[1]], [i, i], color=clrs[curr_betti],
                lw=1.5)
        if curr_betti == dims-1:
            ax.set_xlim([0, largest_pers+0.01])
            ax.set_ylim([-1, len(curr_bar)])
            ax.set_yticks([])
        else:
            ax.set_xlim([0, largest_pers+0.01])
            ax.set_xticks([])
            ax.set_ylim([-1, len(curr_bar)])
            ax.set_yticks([])


''' Draws circles around the points of a point cloud, first dimension contains the number of points '''

def rips_plot(pcloud,radius,graph=False,dmat=None,polygons=False,circles=True):
    plt.plot(pcloud[:,0],pcloud[:,1],'b.')
    fig = plt.gcf()
    ax = fig.gca()  
    for i in range(len(pcloud)):
        if circles == True:
            circle = plt.Circle((pcloud[i,0],pcloud[i,1]),radius,color='r',alpha = 0.025)
            ax.add_artist(circle)
        if graph == True:
            for j in range(len(pcloud)):
                if dmat[i,j]<=radius:
                    if i<j:
                        ax.plot([pcloud[i,0],pcloud[j,0]],[pcloud[i,1],pcloud[j,1]],'k',alpha=0.5)
                if polygons==True:
                    for k in range(len(pcloud)):
                        if dmat[i,j]<=radius and dmat[i,k]<=radius and dmat[j,k]<=radius:
                            polygon = Polygon(pcloud[[i,j,k],:])
                            p = PatchCollection([polygon],alpha=0.5)
                            p.set_array(np.array([5,50,100]))
                            ax.add_collection(p)
    return fig,ax
  

def direct_graph_plot(pcloud,radius,graph=False,dmat=None,polygons=False):
    fig = plt.gcf()
    ax = fig.gca()  
    for i in range(len(pcloud)):
        if graph == True:
            for j in range(len(pcloud)):
                if dmat[i,j]<radius:
                    plt.plot([pcloud[i,0],pcloud[j,0]],[pcloud[i,1],pcloud[j,1]],'k',alpha=0.05)
                    plt.arrow(pcloud[i,0],pcloud[i,1],0.7*(pcloud[j,0]-pcloud[i,0]),0.7*(pcloud[j,1]-pcloud[i,1]),color='b',head_width=0.1,lw=0.01)
                if polygons==True:
                    for k in range(len(pcloud)):
                        if dmat[i,j]<radius and dmat[i,k]<radius and dmat[j,k]<radius:
                            polygon = Polygon(pcloud[[i,j,k],:])
                            p = PatchCollection([polygon],alpha=0.5)
                            p.set_array(np.array([5,50,100]))
                            ax.add_collection(p)



'''Finds the most persistent component of a given dimension > 0'''
def max_pers(pd,dim=1):
    if len(pd[dim])>0:
        pers = pd[dim][:,1] - pd[dim][:,0]
        max_persistence = np.max(pers)
        return max_persistence
    else:
        return 0
    
    
def recurrence_plot(X,thresh=0.1, func = lambda x: pairwise_distances(x)):
    D = func(X)
    D[D>=thresh] = 0
    D[np.logical_and(D<thresh, D!=0)] = 1
    np.fill_diagonal(D,1)
    return D
    
    

def gen_weight_mat(N,rank,g=1,svd=True,eigenvals=[]):
    P = np.zeros([N,N])
    if svd==False:
        for r in range(rank):
            m = np.random.randn(N)
            n = np.random.randn(N)
            P = P + np.outer(m,n)/(1+r)
        gX = ((g**2)/N)*np.random.randn(N,N)
        J = gX + P
        np.fill_diagonal(J,0)
        return J
    elif svd=='eigdecomp':
        U = ortho_group.rvs(N)
        if eigenvals!=[]:
            D = np.diag(np.concatenate([eigenvals,np.zeros(N-rank)]))
        else:
            D = np.diag(np.concatenate([2*np.random.rand(rank)-1,np.zeros(N-rank)]))
        V = np.linalg.inv(U)
        P = 5*rank*(np.matmul(U,np.matmul(D,V)))/N*rank
        gX = ((g**2)/N)*np.random.randn(N,N)
        J = gX + P
        np.fill_diagonal(J,0)
        return J, [U,D,V]
    elif svd=='qr_decomp':
        A = 0.01*np.random.randn(N,N)
        np.fill_diagonal(A,0)
        U = np.linalg.qr(A)
        for i in range(N):
            if i<rank:
                U[1][i,i] = eigenvals[i]
            else:
                U[1][i,i] = 0
        P = U[0]@U[1]@U[0].T
        gX = ((g**2)/N)*np.random.randn(N,N)
        J = gX + P
        # np.fill_diagonal(J,0)
        return J, U
    else:
        U = ortho_group.rvs(N)
        if eigenvals!=[]:
            D = np.diag(np.concatenate([eigenvals,np.zeros(N-rank)]))
        else:
            D = np.diag(np.concatenate([np.sort(np.random.randn(rank)),np.zeros(N-rank)]))
        V = ortho_group.rvs(N)
        P = 5*rank*(np.matmul(U,np.matmul(D,V.T)))/N*rank
        gX = ((g**2)/N)*np.random.randn(N,N)
        J = gX + P
        np.fill_diagonal(J,0)
        return J, [U,D,V]
    
def low_rank_rnn(N,T,I=0,P=0,rank=1,mu=0.05,init_x=[0],g=1,svd=False,act_fun=np.tanh):
    if P is None:
        P = gen_weight_mat(N,rank,g,svd)[0]
    x = np.zeros([N,T])
    if len(init_x)==0:
        x[:,0] = np.random.rand(N)
    else:
        x[:,0] = init_x
    for t in range(T-1):
        dx = -x[:,t]+np.dot(P,act_fun(x[:,t]))+I[:,t]
        x[:,t+1] = x[:,t]+mu*dx
    return x


    
def full_hom_analysis(X,metric='LP',order=False,q=2,pimage=False,dim=2,perm=None,R=0.1,Eps=0.1):
    if metric == 'LP':
        if order==True:
            dmat = order_complex(pairwise_distances(X))
        else:
            dmat = pairwise_distances(X)      
        if order==True:
            dmat = order_complex(Schreiber(X,q))
    elif metric == 'geodesic':
        if order==True:
            dmat = order_complex(geodesic(X,r=R,eps=Eps))
        else:
            dmat = geodesic(X,r=R,eps=Eps)
    hom = tda(dmat,distance_matrix=True,maxdim=dim,n_perm=perm)['dgms']
    return [dmat,hom]


def annotate_imshow(D,round_val=2,txt_size=6):
    fig, ax = plt.subplots(1,1,dpi=200)
    ax.imshow(D,aspect='auto')
    for (j,i),label in np.ndenumerate(D):
        if label!=0:
            ax.text(i,j,round(label,round_val),ha='center',va='center',fontsize=txt_size)

def transient_nets(N,T,W,I,mu=0.05):
    X = np.zeros([N,T])
    for t in range(T-1):
        dX = 10*(-X[:,t]+np.dot(W,X[:,t])+I[:,t])
        X[:,t+1] = X[:,t] + mu*dX
    return X
       

def plot_circl_eig(r):
    rd = np.abs(r)
    r_perc = np.percentile(rd,95)
    theta = np.linspace(0,2*np.pi,100)
    plt.plot(1*np.cos(theta),1*np.sin(theta),'k')
    plt.plot(r_perc*np.cos(theta),r_perc*np.sin(theta),color='g')
    return r_perc
        
   
def get_boundary(x):
    hull = ConvexHull(x)
    points = hull.points
    vertices = hull.vertices
    return np.squeeze(np.array([[points[vertices,0],points[vertices,1]]]))
    
'''
    Regression with RBF kernels:
    x: a pointsXdim array for all the points at which the function is to be evaluated
    c: a list of centers of shape pointsXdim, where they repeat for each entry of points
'''
def RBF_kernels(x,c,eps=1,weights=np.array([1,1])):
    X = np.zeros([x.shape[0],len(c)])
    for i in range(len(c)):
        X[:,i] = np.exp(-eps[i]*((x-c[i])**2)@weights).flatten()
    return X

def RBF_regress(y,x,centers,eps=1,weight=np.array([1,1])):
    X_design = RBF_kernels(x,centers,eps=eps,weights=weight)
    beta = y @ np.linalg.pinv(X_design.T)
    return beta, X_design
    

def gen_mfld(N,mfld_type,*args):    
    if mfld_type=='S1':
        r = args[0]
        theta_S1 = np.linspace(0,2*np.pi,N)
        S1 = np.array([r*np.cos(theta_S1),r*np.sin(theta_S1)])
        return S1
    
    elif mfld_type=='S2':
        r = args[0]
        phi = np.linspace(0,np.pi,N)
        theta = np.linspace(0,2*np.pi,N)
        Phi, Theta = np.meshgrid(phi,theta)
        Phi_S2, Theta_S2 = Phi.flatten(), Theta.flatten()
        S2 = np.array([r*np.cos(Theta_S2)*np.sin(Phi_S2),r*np.sin(Theta_S2)*np.sin(Phi_S2),r*np.cos(Phi_S2)])
        return S2
    
    if mfld_type=='T2':
        R = args[0]
        r= args[1]
        phi = np.linspace(0,2*np.pi,N)
        theta = np.linspace(0,2*np.pi,N)
        Phi, Theta = np.meshgrid(phi,theta)
        Phi_T2, Theta_T2 = Phi.flatten(), Theta.flatten()
        T2 = np.array([(R+r*np.cos(Theta_T2))*np.cos(Phi_T2),(R+r*np.cos(Theta_T2))*np.sin(Phi_T2),r*np.sin(Theta_T2)])  
        return T2

'''Takes in a list of persistence diagrams and calculates the bottleneck distances between them'''
def bottleneck_dmat(pdiag1,pdiag2,dim=1):
    D = np.zeros([len(pdiag1),len(pdiag2)])
    for i in range(len(pdiag1)):
        for j in range(len(pdiag2)):
            if i>j:
                D[i,j] = persim.bottleneck(pdiag1[i][dim],pdiag2[j][dim])
    D = D + D.T
    return D


def bottleneck_time(pdiags,dim=1,features=1,plot=False):
    pers_vals = []
    matchings = []
    for t in range(len(pdiags)-1):
        feat1, feat2 = np.argsort(pdiags[t][dim][:,1]-pdiags[t][dim][:,0])[-features:],np.argsort(pdiags[t+1][dim][:,1]-pdiags[t+1][dim][:,0])[-features:]
        pers_vals.append([pdiags[t][dim][feat1,:], pdiags[t+1][dim][feat2,:]])
        _,matching = persim.bottleneck(np.array(pers_vals[-1][0]),np.array(pers_vals[-1][1]),matching=True)
        matchings.append(matching)
        
    if plot==True:
        for t in range(len(pdiags)-1):
            persim.bottleneck_matching(pers_vals[t][0], pers_vals[t][1], matchings[t])
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.plot([0,1],[0,1],'k')
    return pers_vals, matchings
        
    
    
    
    
    
    
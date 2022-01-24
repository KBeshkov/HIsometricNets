#implementation of algorithms
from sklearn.manifold import Isomap
import numpy as np
import numpy.matlib as mat
import math
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d import Axes3D 
from scipy.ndimage import convolve
import numpy.matlib
from scipy import stats
from scipy.cluster.hierarchy import dendrogram
from scipy.stats import wasserstein_distance, rankdata, entropy, ks_2samp
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import cdist, jensenshannon
from scipy.optimize import linear_sum_assignment
# import PersistenceImages.persistence_images as pimg
from sklearn.neural_network import MLPRegressor
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
from scipy import linalg
from scipy.special import legendre, sph_harm, eval_legendre
from cmath import sqrt as isqrt
from statsmodels.stats.multitest import fdrcorrection
import pickle
import sys
# sys.path.append('C:\\Kosio\\Master_Thesis\\Code\\backup\\')
sys.path
# sys.path.append('E:/Code/CURBD-master/')
# sys.path.append('C:/Kosio/Master_Thesis/Çode/backup/')
# import curbd
from scipy.spatial import ConvexHull
#from pyemd import emd_samples



def open_data(sess,stim_condition,region,bin_step,bin_strt=0,snr=2,frate=1):
    stim = sess.stimulus_presentations[
        sess.stimulus_presentations['stimulus_name'] == stim_condition].index.values
#    stim = sess.stimulus_presentations[np.logical_and(
#        sess.stimulus_presentations['stimulus_name'] == stim_condition,sess.stimulus_presentations[
#                'phase']=='0.0')].index.values
    decent_snr_unit_ids = sess.units[np.logical_and(sess.units['snr'] >= snr, sess.units['firing_rate'] > frate)]        
    decent_snr_unit_ids = list(decent_snr_unit_ids[decent_snr_unit_ids['ecephys_structure_acronym']==region].index.values)
    durations = sess.stimulus_presentations.duration[stim]
    bins = []
    for i in range(len(durations)):
        bins.append(np.arange(bin_strt,np.max(durations.values),bin_step))
    stim_labels = list(sess.stimulus_presentations.stimulus_condition_id[stim])
    spikes_per_stim = []
    count = 0
    for i in np.unique(stim_labels):
        spike_counts_da = sess.presentationwise_spike_counts(
            bin_edges=bins[count],
            stimulus_presentation_ids=stim[stim_labels==i],
            unit_ids=decent_snr_unit_ids,
            binarize=False
        )
        
        spikes_per_stim.append(spike_counts_da.data)
        count = count + 1
    return [spikes_per_stim,durations]



''' 
    Spike count
    -----------
    Input: an array of cellsXtimes with spike times (spike times should be non-zero)
    bin_int: the time interval over which to count spikes
    overlap: if True, lets bins overlap
    
    Output: an array of cellsXspike counts
'''
def spike_count(X,bin_int,overlap=False):
    if overlap == False:
        lngth = int(np.ceil((np.max(X))/bin_int))
        Y = np.zeros([len(X[:,0]),lngth])
        for i in range(len(Y[0,:])):
            Y[:,i] = np.sum(np.logical_and(np.logical_and(X>=i*bin_int, X<=(i+1)*bin_int),X!=0),1)
        return Y
    if overlap == True:
        lngth = int(2*np.ceil((np.max(X)/bin_int)))
        Y = np.zeros([len(X[:,0]),lngth])
        for i in range(len(Y[0,:])):
            Y[:,i] = np.sum(np.logical_and(np.logical_and(X>=i*bin_int-i*bin_int/2,X<=(i+1)*bin_int-i*bin_int/2),X!=0),1)
        return Y
    
    
'''
    Lp metric
    ---------
    Mainly to be done on spike counts
    
    Input: an array of cellsXspike counts
    p: a positive real number to specify the type of Lp metric
    
    Output: a cellsXcells distance matrix
'''
def Lp_metric(X,p=2):
    D = np.zeros([len(X[:,0]),len(X[:,0])])
    for i in range(len(X[:,0])):
        for j in range(len(X[:,0])):
            if  i < j:
                if p>=1:
                    D[i,j] = (np.sum(np.abs(X[i,:]-X[j,:])**p))**(1/p)
                else:
                    D[i,j] = (np.sum(np.abs(X[i,:]-X[j,:])**p))

    D = D + D.T
    return D
    
def mean_difference(X):
    N = len(X[:,0])
    D = np.zeros([N])
    for i in range(N):
        v = np.sum(X[i,:])
        for j in range(N):
            D[i] = D[i] - (v - np.sum(X[j,:]))
    return D/(N-1)
        
    
'''
    Schreiber metric
    ----------------
    Makes the most sense on spike trains
    
    Input: an array cellsXspikes
    sigma: the variance parameter for the gaussian kernel
    mode: if set to 'unbounded' it applies a log function to the distance matrix to make it unbounded.
        if set to 'linear' multiplies the distance by 1000, making it "less" bounded - no...
    Output: a cellsXcells distance matrix
'''
def Schreiber(X,sigma,mode = 'standard'):
    X_fix = np.copy(X)
    X_fix[:,0] = 1
    X_fix[:,-1] = 1
    x = np.linspace(-5,5,len(X_fix[0,:]))
    Kern = (1/np.sqrt(2*np.pi*sigma**2))*np.exp(-(x**2)/(2*sigma**2))
    Y = np.zeros([len(X_fix[:,0]),2*len(X_fix[0,:])-1])
    Y_n = np.zeros([len(X_fix[:,0]),2*len(X_fix[0,:])-1])
    for i in range(len(X_fix[:,0])):
        Y[i,:] = np.convolve(X_fix[i,:],Kern)
        Y_n[i,:] = Y[i,:]/np.linalg.norm(Y[i,:])
    C = np.zeros([len(X_fix[:,0]),len(X_fix[:,0])])
    for i in range(len(X_fix[:,0])):
        for j in range(len(X_fix[:,0])):
            C[i,j] = np.correlate(Y[i,:],Y[j,:])/(np.linalg.norm(Y[i,:])*np.linalg.norm(Y[j,:]))
    D = 1 - C
    if mode == 'unbounded':
        D = -np.log(1-D)
    elif mode == 'linear':
        D = 1000*D
    np.fill_diagonal(D,0)
    return [Y_n,D]

#copied from fit_neuron package   
def schreiber_sim(st_0,st_1,bin_width=0.0001,sigma=0.1,t_extra=0.5):
    """
    Computes Schreiber similarity between two spike trains as described in [SS2003]_.

    :param st_0: array of spike times in seconds
    :param st_1: second array of spike times in seconds
    :keyword bin_width: precision in seconds over which Gaussian convolution is computed
    :keyword sigma: bandwidth of Gaussian kernel
    :keyword t_extra: how much more time in seconds after last signal do we keep convolving?

    .. [SS2003] Schreiber, S., et al. "A new correlation-based measure of spike timing reliability."
        Neurocomputing 52 (2003): 925-931.
    """

    smoother_0 = stats.gaussian_kde(st_0,sigma)
    smoother_1 = stats.gaussian_kde(st_1,sigma)
    t_max = max([st_0[-1],st_1[-1]]) + t_extra
    t_range = np.arange(0,t_max,bin_width)
    st_0_smooth = smoother_0(t_range)
    st_1_smooth = smoother_1(t_range)
    sim = stats.pearsonr(st_0_smooth, st_1_smooth)[0]
    return 1-sim

''' 
Takes a list of cells
'''

def schreiber_dist_mat(X,binwdth=0.001,sigm=0.1):
    Ncells = len(X)
    D = np.zeros([Ncells,Ncells])
    for i in range(Ncells):
        for j in range(Ncells):
            if i < j:
                D[i,j] = schreiber_sim(X[i],X[j],bin_width=binwdth,sigma=sigm)
    D = D + D.T
    return(D)


        
    
#copied from fit_neuron package
def victor_purpura_dist(tli,tlj,cost=1):
    """
    d=spkd(tli,tlj,cost) calculates the "spike time" distance
    as defined [DA2003]_ for a single free parameter,
    the cost per unit of time to move a spike.

    :param tli: vector of spike times for first spike train
    :param tlj: vector of spike times for second spike train
    :keyword cost: cost per unit time to move a spike
    :returns: spike distance metric

    Translated to Python by Nicolas Jimenez from Matlab code by Daniel Reich.

    .. [DA2003] Aronov, Dmitriy. "Fast algorithm for the metric-space analysis
                of simultaneous responses of multiple single neurons." Journal
                of Neuroscience Methods 124.2 (2003): 175-179.

    Here, the distance is 1 because there is one extra spike to be deleted at
    the end of the the first spike train:

    >>> spike_time([1,2,3,4],[1,2,3],cost=1)
    1

    Here the distance is 1 because we shift the first spike by 0.2,
    leave the second alone, and shift the third one by 0.2,
    adding up to 0.4:

    >>> spike_time([1.2,2,3.2],[1,2,3],cost=1)
    0.4

    Here the third spike is adjusted by 0.5, but since the cost
    per unit time is 0.5, the distances comes out to 0.25:

    >>> spike_time([1,2,3,4],[1,2,3,3.5],cost=0.5)
    0.25
    """

    nspi=len(tli)
    nspj=len(tlj)

    if cost==0:
        d=abs(nspi-nspj)
        return d
    elif cost==np.Inf:
        d=nspi+nspj;
        return d

    scr = np.zeros( (nspi+1,nspj+1) )

    # INITIALIZE MARGINS WITH COST OF ADDING A SPIKE

    scr[:,0] = np.arange(0,nspi+1)
    scr[0,:] = np.arange(0,nspj+1)

    if nspi and nspj:
        for i in range(1,nspi+1):
            for j in range(1,nspj+1):
                scr[i,j] = min([scr[i-1,j]+1, scr[i,j-1]+1, scr[i-1,j-1]+cost*abs(tli[i-1]-tlj[j-1])])

    d=scr[nspi,nspj]
    return d


''' 
Takes a list of cells
'''

def VP_dist_mat(X,cst=1):
    Ncells = len(X)
    D = np.zeros([Ncells,Ncells])
    for i in range(Ncells):
        for j in range(Ncells):
            if i != j:
                D[i,j] = victor_purpura_dist(X[i],X[j],cost = cst)
    return(D)
    
def correlation_metric(X):
    Y = np.zeros([len(X),len(X)])
    for i in range(len(X)):
        for j in range(len(X)):
            if i>j:
                Y[i,j] = 1 - np.corrcoef(X[i,:],X[j,:])[0,1]
    Y = Y + Y.T
    return Y
    
'''
    Overlap "metric"
    ----------------
    Computes a "distance matrix" based on the overlap between spikes. This method
    has the advantage of not counting spikes that don't overlap as contributing
    to the difference between receptive fields. It starts with a distance matrix
    where the distances are determined to be 1 + the spike count of the strongest firing cell.
    After that for each overlap between 2 cells 1 is subtracted from that matrix.
    This guarantees that the distances are all positive
    
    Input: an array of cellsXspikes
    
    Output: a distance matrix
    
'''

def overlap_measure(X):
    max_spikes = np.max(np.sum(X,1))
    D = np.ones([len(X[:,0]),len(X[:,0])])
    for i in range(len(X[0,:])):
        overlaps = np.where(X[:,i] >= 1)[0]
        if len(overlaps)>1:
            for k in overlaps:
                for l in overlaps:
                    if k != l:
                        D[k,l] = D[k,l] - 5/max_spikes - 0.000001*np.random.rand(1)
    np.fill_diagonal(D,0)
    return D

#'''
#    Overlap "metric"
#    ----------------
#    Homology can be calculated only when the receptive fields of the cells are
#    contractible.
#    
#    Input: an array of cellsXspike counts
#    stat_thrsh: if 'abs', use an absolute threshold to build simplices
#                between cells. If it is 'sign' calculate a significant threshold
#                for each cell.
#    thrsh_val: value of threshold
#    distrib: if True return the distribution of words which to consider
#    percentile: the percentile of words which to include
#
#    Output: a dictionary containing all the simplices. (This will require
#    writing a function that computes homology without a metric, so 
#    maybe consider outputing the boundary matrices instead, 
#    or a distance matrix with the edge weight sum metric, the one used in Curto & Itskov 2008) 
#'''

def spike_overlap(X,distrib=True, percentile=0.01):
    X = X.astype(int)
    word_distrib = np.zeros(len(X[0,:]))
    for i in range(len(X[0,:])):
        X[X[:,i]>0,i] = 1
        wrd = np.array2string(X[:,i],separator='')
        wrd=wrd.replace('\n','')
        wrd=wrd.replace(' ','')
        word_distrib[i] = (int(wrd[1:-1],2))
    word_distrib = word_distrib[word_distrib!=0]
    wrd_vals = np.unique(word_distrib)
    wrd_smplx = []
    wrd_distrib = []
    for j in range(len(wrd_vals)):
        wrd_fq = np.sum(word_distrib==wrd_vals[j])
        wrd_distrib.append(wrd_fq)
        if wrd_fq>percentile:
            wrd_smplx.append(bin(int(wrd_vals[j]))[2:])
    return [wrd_smplx,np.log2(wrd_vals),wrd_distrib]
    
    
''' 
    Distance matrix of a simplicial complex
    ---------------------------------------
    Outputs a distance matrix corresponding to a given simplicial complex
    
    Input: A spiking sequence of cellsXspikes or a simplicial complex in 
    the form of a list of binary words
    
    Output: an NxN matrix where N is the number of neurons. The distance
    is defined as d(x_i,x_j)= 2 - s(n), where s is a convergent sequence
    and n is the number of times in which x_i and x_j appear together
    in a word. Also d(x_i,x_j) = 0 when i=j
'''
def simp_dist_mat(X,mode='spiking'):
    N = len(X[:,0])
    T = len(X[0,:])
    D = 2*np.ones([N,N])
    Hist = np.ones([N,N])
    for i in range(T):
        for j in range(N):
            for k in range(N):
                if X[j,i] >= 1 and X[k,i] >= 1 and j!=k:
                    D[j,k] = D[j,k] - 1/Hist[j,k]**2
                    Hist[j,k] = Hist[j,k] + 1
    np.fill_diagonal(D,0)
    return D
    
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


''' Perstience landscapes:

    Input:
        X: a birth/death diagram
        
    Output: a dictionary of persistence lanscapes    
'''
#
#def persistence_landscapes(X,k):                
#    Bcurves = []
#    finite_dgm = X[0][np.isfinite(X[0])]
#    x_maxs = []
#    for i in range(len(X)):
#        if len(X[i])>0 and i!=0:
#            x_maxs.append(np.max(X[i]))
#        else:
#            x_maxs.append(np.max(finite_dgm))
#    x_max  = max(x_max)
#    M = {}
#    H = {}
#    for i in range(len(X)):
#        M[str(i)] = []
#        H[str(i)] = []
#        for j in range(len(X[i])):
#            m = (X[i][j][:,0] + X[i][j][:,1])/2
#            h = (X[i][j][:,1] - X[i][j][:,0])/2
#            M[str(i)].append(m)
#            H[str(i)].append(h)
#    duration = 1000
#    for i in range(duration):
#        lambda_kt = 0 
#        for j in range(len(X)):
#            if H[str(j)]
#            lambda_kt = 
            
    
def w(x,a,b,c = 0,d = 0,ftype = 'Mex hat'):
    if ftype == 'Mex hat':
        Weights = a*np.exp(-(x**2)/b)-b*np.exp(-(x**2)/a)
        Weights[x==0] = 0
    elif ftype == 'Mex hat 2.0':
        Weights = a*np.exp(-(x**2)/c)-b*np.exp(-(x**2)/d)
        Weights[x==0] = 0        
    elif ftype == 'Wiz hat':
        Weights = (1-np.abs(x))*np.exp(-np.abs(x))
        Weights[x==0] = 0
    elif ftype == 'Assym Mex hat':
        Weights = (a*np.exp(-((x-c)**2)/b)-b*np.exp(-((x-c)**2)/a))/(a-b)
    elif ftype == 'Gaussian':
        Weights = a*np.exp(-(x**2)/b)
    elif ftype == 'Gaussian_inhib':
        Weights = -a*np.exp(-(1/x**2)*b)
    elif ftype == 'Oscill':
        Weights = a*np.sin(b*x)
    elif ftype == 'Wizzard_hat':
        Weights = (a*np.exp(-(x**2)/b) - b)/(a-b)
    elif ftype == 'Avalanche':
        zero = np.where(x==0)[0][0] #find the index of the current neuron
        Weights = a*np.exp(-x/b)        
        if zero < c:
            Weights[(zero+c)%len(x):] = -1#a*np.exp(-(b/(x[(zero+5)%len(x):])**2))-a*np.exp(-(b/((x[(zero+5)%len(x):])**2+0.5)**2))
            Weights[(zero-c)%len(x):] = -1#0
            Weights[:zero] = -1#0
        elif zero > len(x)-c-1:
            Weights[(zero+c)%len(x):(zero-3)] = -1#a*np.exp(-(b/(x[(zero+5)%len(x):(zero-5)])**2))-a*np.exp(-(b/((x[(zero+5)%len(x):(zero-5)])**2+0.5)**2))
            Weights[(zero-c):zero] = -1#0
        else:
            Weights[(zero+c)%len(x):] = -1#a*np.exp(-(b/(x[(zero+5)%len(x):])**2))-a*np.exp(-(b/((x[(zero+5)%len(x):])**2+0.5)**2))
            Weights[:zero] = -1#a*np.exp(-(b/(x[:zero])**2))-a*np.exp(-(b/((x[:zero])**2+0.5)**2))
            Weights[(zero-c)%len(x):zero] = -1#0
        Weights[zero] = 0
    elif ftype == 'Top strat':
        Weights = np.zeros(np.shape(x))
        for i in range(len(x)):
            for j in range(len(x)):
                if x[i,j] < c:
                    Weights[i,j] = a
                elif x[i,j] > c:
                    Weights[i,j] = -b
        np.fill_diagonal(Weights,0)
    return Weights

    
def F(x,a,b = 0,ftype='sigmoid'):
    if ftype == 'logistic':
        f = 1/(1+np.exp(-a*x))
        return f
    if ftype == 'sigmoid':
        f = 1/(1+np.exp(-a*x+b))
        return f
    elif ftype == 'step':
        f = np.copy(x)
        f[x>0] = 1
        f[x<=0] = 0
        return f
    elif ftype == 'rectify':
        x[x<a] = 0
        return x    
    elif ftype == 'tanh':
        f = 2/(1+np.exp(-2*x))-1
        return f
    
def connectedness(X):
    d = pairwise_distances(X)
    hom = tda(d,distance_matrix=True,maxdim=0)['dgms'][0]
    return hom[-2,1]


'''Calculates the smallest geodesic distance between two points on a Riemannian manifold

    Input:
        g: The metric tensor of the manifold
        init_point: the first point on the manifold
        final_point: the second point on the manifold
'''
def geodesic_dist(X,R,shape = 'S1'):
    D = []
    if shape == 'S1':
        for i in range(len(X)):
            for j in range(len(X)):
                p = np.matlib.repmat(np.array(X[j]).T,len(X),1)
                D.append(np.abs((R**2)*p-(R**2)*X))
                D[-1][D[-1]>R**2*np.pi] = 2*R**2*np.pi-D[-1][D[-1]>R**2*np.pi]
    return(D)

def count_neighbors(D,rad='auto'):
    if rad == 'auto':
        rad = np.percentile(D.flatten(),5)
        Nbrs = np.sum(D<rad,0)-1
    else:
        Nbrs = np.sum(D<rad,0)-1
    return Nbrs
    


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



#                
    
'''Test the convexity of a given receptive field

    Input: An array with corresponding to a receptive field
    
    Output: A real number describing how convex the receptive field is.
'''

def RF_convexity2D(M): 
    d = np.zeros([np.shape(M)[0],np.shape(M)[0]])
    for i in range(np.shape(M)[0]):
        for j in range(np.shape(M)[0]):
            if i > 0 and j > 0 and j < (len(M[:,0])-1) and i < (len(M[:,0])-1):
                d[i,j] = np.linalg.norm(np.ones([3,3])*M[i,j]-M[i-1:i+2,j-1:j+2])
    return np.sum(d)
    
def norm_mat(X):
    mat_max = np.max(X)
    mat_min = np.min(X)
    X = (X-mat_min)/mat_max
    return(X)
    
    
'''Calculates the homology groups of a spike pattern

    Input: 
    X: A list of spike times for a each cell for N stimulus presentations
    mxdims: The number of dimensions for which to calculate the homology groups
    dist_type: The type of distance matrix to compute (default is Schreiber)
    vspace_type: The type of vector space in which the persistence diagrams are mapped to
    sgm: the sigma value of the distance matrix (could be cost for VP, width for Schreiber or p value for Lp metircs)
    
    Output: A list of persistance values for each stimulus and betti curves.
    
'''
def Homology_per_stim(X,mxdims=2,dist_type='Schreiber',vspace_type='Persistence images',sgm=0.05):
    N_presentations = len(X)
    x_epsilon = []
    betti_vectors = []
    betti0_vectors = [] #make this into a dictionary
    betti1_vectors = []
    betti2_vectors = []
#    betti3_curves = []
    betti_pers = []
    betti0_pers = []
    betti1_pers = []
    betti2_pers = []
    for i in range(N_presentations):
        if dist_type=='Schreiber':
            Dist = order_complex(schreiber_dist_mat(X[i],sigm=sgm))
        elif dist_type=='Lp':
            Dist = order_complex(Lp_metric(X[i,:,:].T,p=sgm))
        elif dist_type=='Schreiber_discrete':
            Dist = order_complex(Schreiber(X[i,:,:].T,sigma=sgm)[1])
        if len(Dist[0,:])>100:
            bd_diagram = tda(Dist,distance_matrix=True,maxdim=mxdims,n_perm=75)['dgms']
        else:
            bd_diagram = tda(Dist,distance_matrix=True,maxdim=mxdims)['dgms']
        if vspace_type == 'Betti curves':
            x_epsilon.append(Betti_curves(bd_diagram)[0])
            b_curves = Betti_curves(bd_diagram)[1]
            b_pers = Betti_hist(bd_diagram)[0]
            betti0_vectors.append(b_curves[0])
            betti1_vectors.append(b_curves[1])
            betti2_vectors.append(b_curves[2])
    #        betti3_curves.append(b_curves[3])
            betti0_pers.append(b_pers[0])
            betti1_pers.append(b_pers[1])
            betti2_pers.append(b_pers[2])
        elif vspace_type == 'Persistence images':
            p_imager = pimg.PersistenceImager(pixel_size=0.005,kernel_params={'sigma':np.array([[0.0001,0],[0,0.0001]])})
            p_imager.weight = pimg.weighting_fxns.persistence
            p_imager.weight_params = {'n': 2}
#            pimger.birth_range = (0,0.5)
#            pimger.pers_range = (0,0.5)
            b_images0 = p_imager.transform(bd_diagram[0][:-1,:])
            b_images1 = p_imager.transform(bd_diagram[1])
            b_images2 = p_imager.transform(bd_diagram[2])
            betti0_vectors.append(b_images0.flatten())
            betti1_vectors.append(b_images1.flatten())
            betti2_vectors.append(b_images2.flatten())
            bettin_vectors = np.hstack((betti0_vectors[i],betti1_vectors[i],betti2_vectors[i]))
            betti0_pers = []
            betti1_pers = []
            betti2_pers = []
        print(i)
        betti_vectors.append(bettin_vectors)
    #    betti_curves.append(betti3_curves)
        betti_pers.append(betti0_pers)
        betti_pers.append(betti1_pers)
        betti_pers.append(betti2_pers)
    return [betti_vectors,betti_pers]

            
            
            
            

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
    
    
def plot_tunning_heatmap(x,y):
    H = np.zeros([len(x),len(y)])
    for i in range(len(x)):
        for j in range(len(y)):
            H[i,j] = x[i]+y[j]
    plt.figure()
    plt.imshow(H)
    
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

    
    
def fake_data(N,T,frate,bins):
    X = np.zeros([bins,N,T])
    for b in range(bins):
        for i in range(N):
            rand_v = np.sort(np.random.choice(np.linspace(0,T-1,T),size=frate[i],replace=False)).astype(int)
            X[b,i,rand_v] = 1
    return X
    
    
def diagram_entropy_maximization(diag,sigmas = [0.00000000000001,0.0001,0.001],res=0.005):
    pers = diag[:,1]-diag[:,0]
    entropies = []
    for i in sigmas:
        pimger = pimg.PersistenceImager(birth_range=(0,np.max(diag[:,0])),pers_range=(0,np.max(pers)),
                                pixel_size=res,kernel_params = {'sigma':np.array([[i,0],[0,i]])})
        pimger.weight = pimg.weighting_fxns.persistence
        pimger.weight_params = {'n': 2}
        pimage = pimger.transform(diag,skew=True)
        pimg_pdf = np.histogram(pimage.flatten(),bins = int(np.sqrt(len(pimage))),density = True)
        bin_width = pimg_pdf[1][1]-pimg_pdf[1][0]
        pimg_probs = bin_width*pimg_pdf[0]
        probs_pdf = np.histogram(pimg_probs,bins = 100,density = True)
        pbin_width = probs_pdf[1][1]-pimg_pdf[1][0]
        entropies.append(entropy(pbin_width*probs_pdf[0]))
    return [sigmas,entropies,pimage,probs_pdf]
        
    
def weighted_topological_complexity(diagram):
    pers = []
    for  i in range(len(diagram)-1):
        pers.append(np.sum(diagram[i+1][:,1]-diagram[i+1][:,0]))
    return sum(pers)


''' Permutation test using the Monte-Carlo method based on the sum of L2 distances between two 
    sets of multidimensional vectors in R^n'''

def mc_perm_test(X,Y,nperms):
    xlen,ylen = len(X),len(Y)
    D_true = np.zeros([xlen,ylen])
    for i in range(xlen): #compute the true distance based test statistic
        for j in range(ylen):
            D_true[i,j] = np.sqrt(np.sum(X[i]-Y[j])**2)
    D_stat = np.sum(D_true)
    Z = np.concatenate([X,Y])
    p_val = 0
    for k in range(nperms): #permute the entries of the concatenated array
        perm_idx = np.random.permutation(xlen+ylen)
        Z = Z[perm_idx,:]
        D_shuff = np.zeros([xlen,ylen])
        for i in range(xlen):
            for j in range(ylen):
                D_shuff[i,j] = np.sqrt(np.sum(Z[i,:]-Z[xlen+j,:])**2)
        if np.sum(D_shuff)>D_stat:
            p_val = p_val + 1
    return [D_stat,p_val/nperms]
    

'''subsample array'''
def subsample(X,percent=0.05):
    resamp_X = np.copy(X)
    sub_idx = np.random.choice(np.arange(0,len(X)),int(len(X)*percent))
    resample = np.random.choice(np.arange(0,len(X)),int(len(X)*percent)).astype(int)
    resamp_X[sub_idx] = resamp_X[resample]
    return resamp_X

    
def subsamp_remove(X,n_removed):
    subsamp_X = np.random.permutation(X)[:-n_removed]
    return subsamp_X

''' Bootstrapping the mean with replacement for estimating confidence intervals '''

def bootstrap_ci(X,nsamp):
    param_pdf = np.zeros(nsamp)
    dat_param = np.mean(X)
    for i in range(nsamp):
        resample = np.random.choice(np.arange(0,len(X)),len(X)).astype(int)
        param_pdf[i] = np.mean(X[resample])
    ci1 = np.percentile(param_pdf,5)
    ci2 = np.percentile(param_pdf,95)
    return [dat_param,(ci1,ci2)]
        
def permute_mult(X):
    Y = np.zeros([len(X),len(X[0,:])])
    for i in range(len(X)):
        Y[i,:] = X[i,np.random.permutation(len(X[0,:]))]
    for i in range(len(X[0,:])):
        Y[:,i] = Y[np.random.permutation(len(X)),i]
    return Y

def perm_test_multdim(X,nperm,mean_d=False):
    N = len(X)
    M = len(X.T)
    if mean_d==False:
        real_dist = pairwise_distances(X)
        d_distrib = np.zeros([N,N,nperm])
        for i in range(nperm):
            Y = np.zeros([N,M])
            for j in range(M):
                Y[:,j] = X[np.random.permutation(N),j]
            d_distrib[:,:,i] = pairwise_distances(Y)
        sign_val = np.zeros([N,N])
        for i in range(N):
            for j in range(N):
                if i>j:
                    sign_val[i,j] = (100-percentileofscore(d_distrib[i,j,:].flatten(),real_dist[i,j]))/100
        return sign_val, d_distrib
    else:
        real_dist = np.sum(pairwise_distances(X),0)
        d_distrib = np.zeros([N,nperm])
        for i in range(nperm):
            Y = np.zeros([N,M])
            for j in range(M):
                Y[:,j] = X[np.random.permutation(N),j]
            d_distrib[:,i] = np.sum(pairwise_distances(Y),0)
        sign_val = np.zeros(N)
        for i in range(N):
            sign_val[i] = (100-percentileofscore(d_distrib[i,:].flatten(),real_dist[i]))/100
        return sign_val, d_distrib


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

def plot_nsphere(S,ax_lbls=[]):
    dim = len(S)
    nplots = math.factorial(dim)/(math.factorial(3)*(math.factorial(dim-3)))
    print(nplots)
    nrows = int(nplots/2)
    ncols = int(nplots/nrows)
    count = 0
    fig = plt.figure(dpi=200)
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                if i < j and i < k and j < k:
                    plt.subplot(nrows,ncols,count+1)
                    ax = fig.add_subplot(nrows,ncols,count+1,projection='3d')
                    ax.scatter(S[i,:],S[j,:],S[k,:],s=0.1)
                    ax.set_xlabel(ax_lbls[i])
                    ax.set_ylabel(ax_lbls[j])
                    ax.set_zlabel(ax_lbls[k])
                    ax.grid('off')
                    count = count + 1
    plt.tight_layout()
    
    
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
    
    
    
    
def IF_model(W,T,I,thresh=0,eq=-60,reset=-70,mu=0.05,epsp=5):
    X = np.zeros([len(W),T])
    spikes = np.zeros([len(W),T])
    for t in range(T-1):
        dx = eq-X[:,t]+I[:,t]+np.dot(W,epsp*spikes[:,t])
        X[:,t+1] = X[:,t]+mu*dx
        spikes[np.where(X[:,t+1]>thresh),t+1] = 1
        X[np.where(X[:,t+1]>thresh),t+1] = reset
    return [spikes,X]


def neural_mfld(W,T,I,sfa_rate=1,sfa_str=1,a_sig=5,b_sig=10,mu=0.05,voltage=False):
    N = len(W[0,:])
    M = np.zeros([N,T])
    v = np.zeros([N,T])
    noise = 1*np.random.randn(N,T) - 0.75
    spikes_M = np.zeros([N,T]) #define aray to store the spikes
    if voltage==False:
        for t in range(T-1):
            bernoulli_process = np.random.rand(N) #We count a cell as spiking if it's value is higher than a random sample from a Bernoulli process
            spikes_M[np.where(M[:,t]>bernoulli_process),t] = 1
            M[np.where(M[:,t]<0),t] = 0
            f = -M[:,t] - sfa_str*v[:,t]+ F(np.dot(W,M[:,t])+I[:,t],a_sig,b_sig,ftype='sigmoid')+noise[:,t]#np.dot(W,F(S2[:,t],a_sig,b_sig,ftype='sigmoid'))+I[:,t]+noise[:,t]#
            g = sfa_rate*(M[:,t]-v[:,t])
            k2 = -(M[:,t] + mu*f/2) - sfa_str*(v[:,t]+mu*g/2)+ F(np.dot(W,(M[:,t] + mu*f/2))+I[:,t],a_sig,b_sig,ftype='sigmoid')+noise[:,t] #np.dot(W,F(S2[:,t]+mu*f/2,a_sig,b_sig,ftype='sigmoid'))+I[:,t]+noise[:,t]#
            l2 = sfa_rate*(M[:,t]+mu*f/2-(v[:,t] + mu*g/2))
            k3 = -(M[:,t] + mu*k2/2) - sfa_str*(v[:,t] + mu*l2/2)+ F(np.dot(W,(M[:,t] + mu*k2/2))+I[:,t],a_sig,b_sig,ftype='sigmoid')+noise[:,t]#np.dot(W,F(S2[:,t]+mu*k2/2,a_sig,b_sig,ftype='sigmoid'))+I[:,t]+noise[:,t]#
            l3 = sfa_rate*(M[:,t]+mu*k2/2-(v[:,t] + mu*l2/2))
            k4 = -(M[:,t] + mu*k3) - sfa_str*(v[:,t] + mu*l3)+ F(np.dot(W,(M[:,t] + mu*k3))+I[:,t],a_sig,b_sig,ftype='sigmoid')+noise[:,t]#np.dot(W,F(S2[:,t]+mu*k3/2,a_sig,b_sig,ftype='sigmoid'))+I[:,t]+noise[:,t]#
            l4 = sfa_rate*(M[:,t]+mu*k3-(v[:,t] + mu*l3))
            M[:,t+1] = M[:,t] + (mu/6)*(f+2*k2+2*k3+k4)
            v[:,t+1] = v[:,t] + (mu/6)*(g+2*l2+2*l3+l4)
    else:
        for t in range(T-1):
            poisson_process = np.random.poisson(10,size=N) #We count a cell as spiking if it's value is higher than a random sample from a Bernoulli process
            spikes_M[np.where(M[:,t]>poisson_process),t] = 1
            M[np.where(M[:,t]<0),t] = 0
            f = -M[:,t] - sfa_str*v[:,t]+ np.dot(W,F(M[:,t],a_sig,b_sig,ftype='sigmoid'))+I[:,t]+noise[:,t]#np.dot(W,F(S2[:,t],a_sig,b_sig,ftype='sigmoid'))+I[:,t]+noise[:,t]#
            g = sfa_rate*(M[:,t]-v[:,t])
            k2 = -(M[:,t] + mu*f/2) - sfa_str*(v[:,t]+mu*g/2)+ np.dot(W,F((M[:,t] + mu*f/2),a_sig,b_sig,ftype='sigmoid'))+I[:,t]+noise[:,t] #np.dot(W,F(S2[:,t]+mu*f/2,a_sig,b_sig,ftype='sigmoid'))+I[:,t]+noise[:,t]#
            l2 = sfa_rate*(M[:,t]+mu*f/2-(v[:,t] + mu*g/2))
            k3 = -(M[:,t] + mu*k2/2) - sfa_str*(v[:,t] + mu*l2/2)+ np.dot(W,F((M[:,t] + mu*k2/2),a_sig,b_sig,ftype='sigmoid'))+I[:,t]+noise[:,t]#np.dot(W,F(S2[:,t]+mu*k2/2,a_sig,b_sig,ftype='sigmoid'))+I[:,t]+noise[:,t]#
            l3 = sfa_rate*(M[:,t]+mu*k2/2-(v[:,t] + mu*l2/2))
            k4 = -(M[:,t] + mu*k3) - sfa_str*(v[:,t] + mu*l3)+ np.dot(W,F((M[:,t] + mu*k3),a_sig,b_sig,ftype='sigmoid'))+I[:,t]+noise[:,t]#np.dot(W,F(S2[:,t]+mu*k3/2,a_sig,b_sig,ftype='sigmoid'))+I[:,t]+noise[:,t]#
            l4 = sfa_rate*(M[:,t]+mu*k3-(v[:,t] + mu*l3))
            M[:,t+1] = M[:,t] + (mu/6)*(f+2*k2+2*k3+k4)
            v[:,t+1] = v[:,t] + (mu/6)*(g+2*l2+2*l3+l4)
    return [M,v,spikes_M]
    

def Hebbian_rnn_fr(N,T,I,mu=0.1,rule='covariance',wmat=0):
    if len(wmat)==N:
        W = np.copy(wmat)
    elif wmat==0:
        W = 10*np.random.randn(N,N)
    np.fill_diagonal(W,0)
    x = np.zeros([N,T])
    for t in range(T-1):
        dx = -x[:,t] + np.dot(W,np.tanh(x[:,t])) + I[:,t]
        if rule=='Oja':
            for i in range(N):
                for j in range(N):
                    W[i,j] = W[i,j] + mu*(x[i,t]*x[j,t]-W[i,j]*(x[i,t]**2))   
        if rule == 'Hebb':
            dW = np.outer(x[:,t],x[:,t])
            W = W + mu*dW
            W = W/np.max(np.abs(W))
            np.fill_diagonal(W,0)
        x[:,t+1] = x[:,t]+mu*dx
        np.fill_diagonal(W,0)
    return x, W
    
    
#Under construction - need to implement sampling if an Ising model is to be used
#for now we can use a boltzman machine type solution
def Boltzmann_model(W,T,bias = 1,tmp=1):
    X = np.zeros([len(W),T])
    for t in range(T):
        for i in range(len(W)):
            H = np.dot(W[i,:],X[:,t])+bias
            p_x = 1/(1+np.exp(-H/tmp))
            if p_x > np.random.rand():
                X[i,t+1] = 1
    return X

def sparsify(X,p):
    for i in range(len(X)):
        for j in range(len(X)):
            if np.random.rand()<p:
                X[i,j] = 0
    return X

'''Extracts the cells with the strongest tunning curves by minimizing the entropy 
   of  their tunning distribution. Takes a cellsXfeatures matrix.
'''
def extract_tuned_cells(X,norm=False,thresh= 1):
    Y = np.copy(X)
    if norm == False:
        for i in range(len(X)):
            Y[i,:] = X[i,:]/np.sum(X[i,:])
    entropies = np.zeros(len(X))
    for i in range(len(X)):
        entropies[i] = -np.sum(np.multiply(Y[i,:],np.log(Y[i,:])))            
    return [entropies,np.where(entropies<thresh)[0]]

'''Finds the most persistent component of a given dimension > 0'''
def max_pers(pd,dim=1):
    if len(pd[dim])>0:
        pers = pd[dim][:,1] - pd[dim][:,0]
        max_persistence = np.max(pers)
        return max_persistence
    else:
        return 0
    
    
def reccurence_plot(d,thresh=0.1):
    D = np.copy(d)
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
#        np.fill_diagonal(J,0)
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
    
def low_rank_rnn(N,T,I=0,P=0,rank=1,mu=0.05,init_x=[0],g=1,svd=False):
    if P is None:
        P = gen_weight_mat(N,rank,g,svd)[0]
    x = np.zeros([N,T])
    if len(init_x)==0:
        x[:,0] = np.random.rand(N)
    else:
        x[:,0] = init_x
    for t in range(T-1):
        dx = -x[:,t]+np.dot(P,np.tanh(x[:,t]))+I[:,t]
        x[:,t+1] = x[:,t]+mu*dx
    return x


    
def full_hom_analysis(X,metric='LP',order=False,q=2,pimage=False,dim=2,perm=None,R=0.1,Eps=0.1):
    if metric == 'LP':
        if order==True:
            dmat = order_complex(pairwise_distances(X))
        else:
            dmat = pairwise_distances(X)
    elif metric=='correl':
        if order==True:
            dmat = order_complex(correlation_metric(X))
        else:
            dmat = correlation_metric(X,q)        
    elif metric == 'Schreiber':
        if order==True:
            dmat = order_complex(Schreiber(X,q))
        else:
            dmat = Lp_metric(Schreiber(X,q))
    elif metric == 'geodesic':
        if order==True:
            dmat = order_complex(geodesic(X,r=R,eps=Eps))
        else:
            dmat = geodesic(X,r=R,eps=Eps)
    hom = tda(dmat,distance_matrix=True,maxdim=dim,n_perm=perm)['dgms']
    return [dmat,hom]


def temporal_phom(X,tbin=20):
    nhoms = int(len(X[0,:])/tbin)
    thoms = []
    fbins = np.arange(0,int(len(X[0,:])),tbin)
    sbins = np.arange(int(tbin/2),int(len(X[0,:])),tbin)
    count1 = 0
    count2 = 0
    for i in range(2*(nhoms-1)):
        if i%2==0:
            thoms.append(full_hom_analysis(X[:,fbins[count1]:fbins[count1+1]].T,order=False,perm=None)[1])
            count1 = count1 + 1
        else:
            thoms.append(full_hom_analysis(X[:,sbins[count2]:sbins[count2+1]].T,order=False,perm=None)[1])
            count2 = count2 + 1
        print(i)
    full_thom = full_hom_analysis(X.T,order=False,perm=200)[1]
    return [thoms,full_thom]
       
    
def complex_sqrt(x):
    cx = np.zeros(len(x),dtype=np.complex_)
    for i in range(len(x)):
        cx[i] = isqrt(x[i])
    return cx
        
        
def gen_mfld_period(N,R=1,fqs=1,mtype='S1'):
    t = np.linspace(0,2*np.pi,N,endpoint=False)
    if mtype=='S1':
        x = R*np.cos(t)
        y = R*np.sin(t)
        z = []
        for i in range(len(fqs)):
            z.append(R*np.sin(fqs[i]*t))
        return np.vstack((x,y,z))
    elif mtype=='S2':
        t_ = np.linspace(0,np.pi,N)
        t1,t2 = np.meshgrid(t,t_)
        t1,t2 = t1.flatten(),t2.flatten()
        x = R*np.cos(t1)*np.sin(t2)
        y = R*np.sin(t1)*np.sin(t2)
        z = R*np.cos(t2)
        u = []
        for i in range(len(fqs)):
            u.append(spherical_harmonics(fqs[i],0,N,R)[0]*np.sin(fqs[i]*t1))#R*np.cos(fqs[i]*t2)*np.sin(fqs[i]*t1))#R*np.cos(fqs[i]*t2)*np.sin(fqs[i]*t2))##
        return np.vstack((x,y,z,u))
    elif mtype=='sph_harm':
        t_ = np.linspace(0,np.pi,N)
        t1,t2 = np.meshgrid(t,t_)
        t1,t2 = t1.flatten(),t2.flatten()
        
        u = []
        inv_sin = np.zeros(len(t1))
        for i in range(len(t1)):
            if np.sin(t1[i])>0:
                inv_sin[i] = np.sqrt(np.sin(t1[i]))
            if np.sin(t1[i])<0:
                inv_sin[i] = np.sqrt(-np.sin(t1[i]))
        for i in range(len(fqs)):
            u.append(spherical_harmonics(fqs[i],0,N,R)[0]*np.sin(t1))#R*np.cos(fqs[i]*t2)*np.sin(fqs[i]*t2))##
        return np.vstack(u)
    elif mtype=='legendre_functs':
        t = np.linspace(-1,1,N)
        u = []
        for i in range(len(fqs)):
            u.append(eval_legendre(fqs[i],t)/np.sum(np.abs(eval_legendre(fqs[i],t))))
        return np.vstack(u)
        

def mfld_derivs(fqs,angle):
    dx = np.zeros([len(fqs),len(angle)])
    dx2 = np.zeros([len(fqs),len(angle)])
    for i in range(len(fqs)):
        dx[i,:] = fqs[i]*np.cos(fqs[i]*angle)
        dx2[i,:] = (fqs[i]**2)*np.sin((fqs[i]**2)*angle)
    return dx,dx2


def spherical_harmonics(l,m,N,R=1):
    t1 = np.linspace(0,2*np.pi,N)
    t2 = np.linspace(0,np.pi,N)
    theta,phi = np.meshgrid(t1,t2)
    theta, phi = theta.flatten(), phi.flatten()
    x = R*np.cos(theta)*np.sin(phi)
    y = R*np.sin(theta)*np.sin(phi)
    z = R*np.cos(phi)
    Y = sph_harm(m,l,theta,phi)
    if m < 0:
        Y = Y.imag
    elif m >= 0:
        Y = Y.real
    Yx, Yy, Yz = np.abs(Y).T * np.array([x,y,z])
    return Y,np.vstack([Yx,Yy,Yz]).T


def dirac_delta(x):
    if x==0:
        return 1
    else:
        return 0

    
def elipse_2nd_der(theta,a=0):
    return np.squeeze(np.array([[-np.cos(theta)-a*np.sin(theta)],[-np.sin(theta)-a*np.cos(theta)]]))
    

def curvature_per_point(M):
    dims  = len(M.T)
    dx_dt = np.zeros(np.shape(M))
    for i in range(dims):      
        dx_dt[:,i] = np.gradient(M[:,i])
    ds_dt = np.sqrt(np.sum(np.multiply(dx_dt,dx_dt),1))
    tangent = np.array([1/ds_dt]*dims).T*dx_dt
    d_tang = np.zeros(np.shape(M))
    for i in range(dims):
        d_tang[:,i] = np.gradient(tangent[:,i])
    curvature = np.sqrt(np.sum(np.multiply(d_tang,d_tang),1))
    return curvature
    
    
    
def check_mfld_top(X1,X2,reg_ids,compressed=True):
    if compressed == True:
        X_bars = np.zeros(5)
        X_bars[0] = np.sum(np.logical_and(X1[reg_ids]==0,X2[reg_ids]==0))
        X_bars[1] = np.sum(np.logical_and(X1[reg_ids]==1,X2[reg_ids]==0))
        X_bars[2] = np.sum(np.logical_and(X1[reg_ids]==0,X2[reg_ids]==1))
        X_bars[3] = np.sum(np.logical_and(X1[reg_ids]==2,X2[reg_ids]==1))
        X_bars[4] = len(reg_ids)-np.sum(X_bars[:4])
        return X_bars
    else:
        X_bars = np.zeros(16)
        count = 0
        for i in range(4):
            for j in range(4):
                X_bars[count] = np.sum(np.logical_and(X1[reg_ids]==j,X2[reg_ids]==i))
                count = count + 1
        return X_bars
        
    
    
'''Returns a list of the receptive fields of each cell parametrized by variables:
    X is a neuronsXtimes matrix
    L is a stimulus_labelsXtimes matrix with the same shape as X
'''
def find_receptive_field(X,L):
    N = len(X)
    stim_num = np.unique(L)
    rfields = np.zeros([N,len(stim_num)])
    for i in stim_num:
        t_L = np.where(L==i)[0]
        rfields[:,i] = np.mean(X[:,t_L],1)
    return rfields
    
    
            
def full_perm_analysis(dat,dim=2,geod_ep=0.2,metric= 'geod'):
    x_dg_temp = permute_mult(dat)
    if metric=='geod':   
        dmat_temp = geodesic(x_dg_temp,eps=geod_ep)
    elif metric=='LP':
        dmat_temp = pairwise_distances(x_dg_temp)
    hom_dg_temp = normal_bd_dist(tda(dmat_temp,distance_matrix=True,maxdim=dim,n_perm=None)['dgms'])
    hom1 = max_pers(hom_dg_temp,dim=1)
    if dim==2:
        hom2 = max_pers(hom_dg_temp,dim=2)
        return [hom1,hom2]
    else:
        return hom1    
    
def fdr(p_vals):
    ranked_p_values = rankdata(p_vals)
    fdr = p_vals*len(p_vals)/ranked_p_values
    fdr[fdr > 1] = 1
    return fdr

def mfld_bootstrap(h1,h2,region_ids,n_sub,n_perm):
     X = np.zeros([5,5,n_perm])
     for i in range(n_perm):
         rand_sub = np.sort(np.random.choice(len(h1),len(h1)-n_sub,replace=False))
         h1_perm = h1[rand_sub]
         h2_perm = h2[rand_sub]
         region_ids_perm = []
         for j in region_ids:
             reg_permlist = []
             intersect = np.intersect1d(j,rand_sub)
             for r in range(len(intersect)):
                 reg_permlist.append(np.where(intersect[r]==rand_sub)[0][0])
             region_ids_perm.append(reg_permlist)
         LP_bars = check_mfld_top(h1_perm,h2_perm,region_ids_perm[0])
         VISp_bars = check_mfld_top(h1_perm,h2_perm,region_ids_perm[1])
         VISrl_bars = check_mfld_top(h1_perm,h2_perm,region_ids_perm[2])
         VISl_bars = check_mfld_top(h1_perm,h2_perm,region_ids_perm[3])
         VISam_bars = check_mfld_top(h1_perm,h2_perm,region_ids_perm[4])
         X[:,:,i] = np.vstack([LP_bars,VISp_bars,VISrl_bars,VISl_bars,VISam_bars])
     return X
    
    
    
def model_distrib_comp(pd_bar,region_ids,nsamp,nperm,N=5):
    pdns = np.zeros([np.shape(pd_bar)[0],np.shape(pd_bar)[1],nperm])
    pvals_all = np.zeros([np.shape(pd_bar)[0],np.shape(pd_bar)[1],nperm])
    for p in range(nperm):
        for n in range(len(pd_bar)):
            temp_dns = np.random.choice(np.arange(0,N),size=nsamp,p=pd_bar[n])
            for i in range(N):
                pdns[n,i,p] = np.sum(temp_dns==i)
        LP_bars = pdns[0,:,p]
        VISp_bars = pdns[1,:,p]
        VISrl_bars = pdns[2,:,p]
        VISl_bars = pdns[3,:,p]
        VISam_bars = pdns[4,:,p]
        X = np.vstack([LP_bars,VISp_bars,VISrl_bars,VISl_bars,VISam_bars])
        pvals_all[:,:,p] = perm_test_multdim(X,1000)[0]
    return X,pvals_all


def session_bootstrap_test(h1,h2,region_ids,sessions,nperm,perm=True,stat=True):
    n_regions = len(region_ids)
    X = np.zeros([n_regions,5,nperm])
    X_adj = np.zeros([n_regions,5,nperm])
    un_sess = np.unique(sessions)
    nsamp = len(sessions)
    for i in range(nperm):
        rand_sub = np.sort(np.random.choice(un_sess,nsamp))
        h1_temp = []
        h2_temp = []
        temp_regions = [[],[],[],[],[]]
        count = 0
        for r in rand_sub:
            h1_temp.append(h1[sessions==r])
            h2_temp.append(h2[sessions==r])
            sess_reg = np.where(sessions==r)[0]
            for n in range(n_regions):
                if np.sum(np.in1d(sess_reg,region_ids[n]))!=0:
                    temp_regions[n].append(count)
                    count = count+1
        h1_temp = np.hstack(h1_temp)
        h2_temp = np.hstack(h2_temp)
        X_bars = []
        X_bars_adj = []
        for j in range(n_regions):
            X_bars.append(check_mfld_top(h1_temp,h2_temp,temp_regions[j]))
            X_bars_adj.append(check_mfld_top(h1_temp,h2_temp,temp_regions[j])/len(temp_regions[j]))
        X[:,:,i] = np.vstack(X_bars)
        if perm == True:
            xadj = np.vstack(X_bars_adj)
            np.random.shuffle(xadj)
            X_adj[:,:,i] = xadj
        else:
            X_adj[:,:,i] = np.vstack(X_bars_adj)
    if stat==True:
        pvals = np.zeros([5,5])
        for i in range(5):
            for j in range(5):
                if i>j:
                    pvals[i,j] = stats.wilcoxon(X_adj[i,:,:].flatten(),X_adj[j,:,:].flatten())[1]
        return X,pvals,X_adj
    return X_adj
    
    
def cohen_d(x,y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2)/dof)

def Q_stat(x,y):
    D = np.zeros([len(x),len(y)])
    for i in range(len(x)):
        for j in range(len(y)):
            D[i,j] = x[i]-y[j]
    D_med = np.median(D.flatten())
    F = D.flatten()-np.abs(D_med)
    Q = np.sum(F<D_med)/(len(x)*len(y))
    return (Q-0.5)/0.5

def percent_of_score_test(X,Y):
    pvals = np.zeros([np.shape(X)[0],np.shape(X)[1]])
    for i in range(np.shape(X)[0]):
        for j in range(np.shape(X)[1]):
            pvals[i,j] = percentileofscore(X[:,j,:].flatten(),Y[i,j])/100
    return pvals

def stat_image(X,xtick,ytick,save=False,fontsz=14):
    fig, ax = plt.subplots(1,1,dpi=200)
    im1 = ax.imshow(X,cmap='coolwarm')
    for (j,i),label in np.ndenumerate(X):
        ax.text(i,j,round(label,3),ha='center',va='center',fontsize=fontsz)
    ax.set_xticks(np.arange(0,len(X.T)))
    ax.set_yticks(np.arange(0,len(X)))
    ax.set_xticklabels(xtick,fontsize=20,rotation=15)
    ax.set_yticklabels(ytick,fontsize=20)
    if save!=False:
        plt.savefig('C:\Kosio\Master_Thesis\Figures\Paper_top_geod\\'+str(save))

def annotate_imshow(D,round_val=2,txt_size=6):
    fig, ax = plt.subplots(1,1,dpi=200)
    ax.imshow(D,aspect='auto')
    for (j,i),label in np.ndenumerate(D):
        if label!=0:
            ax.text(i,j,round(label,round_val),ha='center',va='center',fontsize=txt_size)


def gabor_filter(x,y,theta,lambd):
    x_p = x*np.cos(theta) + y*np.sin(theta)
    y_p = -x*np.sin(theta) + y*np.cos(theta)
    phi = np.exp(-(x_p**2+y_p**2)/2)*np.cos(2*np.pi*(x_p)*lambd)
    return phi

def drifting_grating(x,y,theta,lambd):
    x_p = x*np.cos(theta) + y*np.sin(theta)
    phi = np.cos(2*np.pi*lambd*x_p)
    return phi

def z_score(x):
    return (x-np.mean(x))/np.std(x)

def check_mfld(h1,h2,thresh=0.2):
    X1 = np.sum((h1[:,1]-h1[:,0])>thresh)
    X2 = np.sum((h2[:,1]-h2[:,0])>thresh)
    X_bars = np.zeros(5)
    X_bars[0] = np.sum(np.logical_and(X1==0,X2==0))
    X_bars[1] = np.sum(np.logical_and(X1==1,X2==0))
    X_bars[2] = np.sum(np.logical_and(X1==0,X2==1))
    X_bars[3] = np.sum(np.logical_and(X1==2,X2==1))
    X_bars[4] = 1-np.sum(X_bars[:4])
    return X_bars

def diff_mfld(X):
    X_diff = X[:,1:]-X[:,:-1]
    return X_diff


def transient_nets(N,T,W,I,mu=0.05):
    X = np.zeros([N,T])
    for t in range(T-1):
        dX = 10*(-X[:,t]+np.dot(W,X[:,t])+I[:,t])
        X[:,t+1] = X[:,t] + mu*dX
    return X
       
def DMD(X):
    V_1 = X[:,:-1]
    V_2 = X[:,1:]
    
    #Calculate the SVD of V_1
    S = np.linalg.svd(V_1,False)
    U = np.copy(S[0])
    V = S[2].conj().T
    D = np.linalg.inv(np.diag(S[1]))
    A = np.dot(np.dot(np.dot(U.conj().T,V_2),V),D)
    eigval, eigvect = np.linalg.eig(A)
    Phi = np.dot(np.dot(np.dot(V_2,V),D),eigvect)
    b = np.dot(np.linalg.pinv(Phi), V_1[:,0])
    t_vect = np.arange(0,len(X))
    Psi = np.zeros([len(V_1.T), len(t_vect)], dtype='complex')
    for i,t in enumerate(t_vect):
        Psi[:,i] = np.multiply(np.power(eigval, t/0.01), b)
    X_pred = np.dot(Phi.T,Psi.T)
    return Phi,A,eigval,eigvect, X_pred
    
#Define tunning curves
def tun_curve(x,theta,amp=1,width=1,ftype='Gaussian'):
    if ftype == 'Gaussian':
        if len(np.shape(x))==1:
            return amp*np.exp(-((x-theta)**2)/2*width**2)
        else:
            return amp*np.exp(-(np.sum((x-theta)**2/2*width**2,1)))
    elif ftype == 'Cos': #width denotes the frequency - it should be set to one
        return np.maximum(amp*np.cos(np.sum(width*x-theta,1)),0)
    elif ftype == 'Circle':
             return amp*np.exp(-(np.sum(np.arctan2(x,theta)**2/2*width**2,1)))
   
#Define Poisson model neurons
#input should be neuronsXtimes or neuronsXtimesXdimensions for multidim manifolds
def Poisson_net(tuning,In,a=1,f='Gaussian',wdth=1):
    N = np.shape(In)[0]
    T = np.shape(In)[1]
    if len(np.shape(In))==2:
        X = np.zeros([N,T])
        for t in range(T):
            X[:,t] = tun_curve(In[:,t],tuning,ftype=f,amp=a,width=wdth)
        P_x = np.random.poisson(X,[N,T])
    else:
        D = np.shape(In)[2]
        X = np.zeros([N,T])
        for t in range(T):
            X[:,t] = tun_curve(In[:,t,:],tuning,ftype=f,amp=a,width=wdth)
        P_x = np.random.poisson(X,[N,T])
    return P_x

def Poisson_receiver(In,W):
    N = len(W)
    M = len(W.T)
    T = len(In.T)
    X = np.zeros([M,T])
    for t in range(T):
        X[:,t] = np.dot(W.T,In[:,t])
    X[X<0] = 0
    P_x = np.random.poisson(X,[M,T])
    return P_x

def sawtooth(x,a,p):
    return -((2*a)/np.pi)*np.arctan(1/np.tan(p*x*np.pi))

def propagate_RNN(init_x,W,inp,mu=0.5):
    X = np.zeros(np.shape(inp))
    X[:,0] = init_x
    for t in range(len(inp.T)-1):
        dX = -X[:,t] + np.dot(W,np.tanh(X[:,t])) + inp[:,t]
        X[:,t+1] = X[:,t] + mu*dX
    return X

def sequence_to_symbols(x):
    sz = np.shape(x)[1]
    uniq_entries = np.unique(x,axis=1)
    sequence = np.zeros(sz)
    for i in range(sz):
        found=False
        c = 0
        while found==False:
            found = np.sum(x[:,i]==uniq_entries[:,c])==len(x)
            c = c + 1
        sequence[i] = c
    return sequence
    
    
def Ziv_Lempel_info(x):
    x = x.astype(int)
    N = len(x)
    words = []
    counter = 0
    while counter<N:
        temp_word = []
        temp_word2 = []
        c = 0
        counter2 = counter
        while temp_word2 not in words and counter<N:
            temp_word.append(x[counter2+c])
            c = c + 1
            counter = counter + 1
            if int("".join(map(str, temp_word))) not in words:
                words.append(int("".join(map(str, temp_word))))
                temp_word2 = words[-1]
    M = len(words)
    return M/N, words


def sol_space_volume(X,W,inp,init_x,mu,res):
    t_ = np.arange(0,1,res)
    T = len(inp[0].T)
    R2 = np.zeros([len(W),len(W),len(t_)])
    for i in range(len(W)):
        for j in range(len(W)):
            if i>j:
                for t in range(len(t_)):
                    W_new = (1-t_[t])*W[i] + t_[t]*W[j]
                    inp_new = inp[i]#(1-t_[t])*inp[i] + t_[t]*inp[j]
                    mu_new = mu[i]
                    if i != 0:
                        X_new = propagate_RNN(init_x,W=W_new,inp=inp_new,mu=mu_new)[:,np.arange(0,2*T-1,2)]
                    else:
                        X_new = propagate_RNN(init_x,W=W_new,inp=inp_new,mu=mu_new)
                    R2[i,j,t] = r2_score(X,X_new)
    return R2
                
                
def perm_invar_dist(X,Y):
    N = len(X)
    d = cdist(X,Y,'jensenshannon')
    rank_d = rankdata(d).reshape(np.shape(d))
    perm_id = np.argsort(np.min(rank_d,0))
    Y_new = np.copy(Y)
    for i in range(len(perm_id)):
        if perm_id[i]>i:
            Y_new[[i,perm_id[i]]] = Y_new[[perm_id[i],i]]
    d_min = (1/N)*np.sum(jensenshannon(np.abs(X),np.abs(Y_new)))
    return d_min, Y_new
                
    
def interpolate_avg(X):
    X_new = np.zeros([np.shape(X)[0],2*np.shape(X)[1]])
    count = 0
    for i in range(len(X_new.T)-1):
        if i%2 == 0:
            X_new[:,i] = X[:,count]
            count = count+1
        else:
            X_new[:,i] = (X[:,count-1]+X[:,count])/2
    return X_new


def frozen_noise(N,T,tau=0.1,mu=0.05):
    h = np.zeros([N,T])
    for t in range(T-1):
        dhdt = (-h[:,t] + np.random.randn(N))/tau
        h[:,t+1] = h[:,t] + mu*dhdt
    return h

def ts_deriv(X,h):
    return (X[:,1:]-X[:,:-1])/h

def dyn_syst_connect_est(Xt,h,alpha=1,method='Lasso'):
    N = len(Xt)
    dXdt  = ts_deriv(Xt,h)
    Y = dXdt+Xt[:,1:]
    X = np.tanh(Xt[:,1:])
    J = np.zeros([N,N])
    if method == 'Lasso':
        reg = Lasso(alpha=alpha)
        for i in range(N):
            X_temp = np.copy(X)
            X_temp[i,:] = 0
            J[:,i] = reg.fit(X_temp.T,Y[i,:]).coef_
    elif method == 'Ridge':
        reg = Ridge(alpha=alpha)
        for i in range(N):
            X_temp = np.copy(X)
            X_temp[i,:] = 0
            J[:,i] = reg.fit(X_temp.T,Y[i,:]).coef_
    elif method == 'pinv':
        for i in range(N):
            J[:,i]=np.dot(Y[i,:],np.linalg.pinv(X))
    return J

def currents_by_region(X,W,regions):
    Currents = []
    for i in range(len(regions)):
        for j in range(len(regions)):
            Currents.append(np.dot(W[np.ix_(regions[i],regions[j])],X[regions[i],:]))
    return Currents
        
def inject_input(N,T,mode='rand',n_pulses=10,mfld=None,eigvect=None,eigval=None):
    X = np.zeros([N,T])
    if mode=='rand':
        return np.random.randn(N,T)
    elif mode=='oscil':
        phase = np.linspace(0,4*np.pi,T)
        count = 0
        for t in range(T):
            X[count:count+10,t] = 100
            count = count + 1
            if count == N-5:
                count = 0
        return X
    elif mode=='oscil2':
        X = 100*np.sin(np.matlib.repmat(np.linspace(0,8*np.pi,T),N,1))
        return X
    elif mode=='pulse':
        p_time = np.random.randint(0,T)
        X[:,p_time] = 100
        return X
    elif mode=='pulses':
        for n in range(n_pulses):
            p_time = np.random.randint(0,T)
            X[:,p_time] = 100
        return X
    elif mode=='ramp':
        st_time = int(T/2)
        for t in range(0,int(T/2)-1):
            X[:,st_time+t+1] = X[:,st_time+t] + 1 
        return X
    elif mode=='eigendir':
        realv = np.where(np.imag(eigval)==0)[0]
        eigmat = eigvect[:,realv[:len(mfld.T)]]
        X = np.dot(mfld,eigmat.T).T
        return np.real(X)
            
def stim_traj_plot(X,s_times,plot=True):
    traj_list = []
    full_PCA = PCA(n_components=2).fit_transform(X.T)
    for i in range(len(s_times)-1):
        traj_list.append(PCA(n_components=2).fit_transform(X[:,s_times[i]:s_times[i+1]].T))
    if plot==True:
        plt.figure(dpi=200)
        for i in range(len(traj_list)):
            plt.subplot(2,1,1)
            plt.plot(full_PCA[:,0],full_PCA[:,1],'k-o')
            plt.subplot(2,1,2)
            plt.plot(traj_list[i][:,0],traj_list[i][:,1],'-o')
            plt.text(traj_list[i][i,0],traj_list[i][i,1],'stim '+str(i))
    return traj_list

def explore_rnn(init_x,T,W,I,epsilon=1e-3,iters=100,reduced=True,fit_rnn=False,g_=5):
    traj_list = []
    full_traj = []
    RNN_dict = {}
    RNN_dict['traces'] = []
    RNN_dict['weights'] = []
    RNN_dict['inputs'] = []
    RNN_dict['traces_pca'] = []
    for i in range(iters):
        if reduced==True:
            X_net = low_rank_rnn(len(init_x),T,I,W,init_x=init_x+epsilon*np.random.randn()).T
            # X_net = X_net/np.max(np.abs(X_net))
            full_traj.append(X_net)
            traj_list.append(PCA(n_components=2).fit_transform(full_traj[i]))
            if fit_rnn==True:
                RNN = curbd.trainMultiRegionRNN(X_net.T,dtData=0.1,dtFactor=2, g=g_,tauRNN=1, P0=1, tauWN=0.1,nRunTrain=20, verbose=False,plotStatus=False,nRunFree=5)
                RNN_interp = RNN['RNN'][:,np.arange(0,2*T,2)]
                RNN_dict['traces'].append(RNN_interp.T)
                RNN_dict['traces_pca'].append(PCA(n_components=2).fit_transform(RNN_interp.T))
                RNN_dict['weights'].append(RNN['J'])
                RNN_dict['inputs'].append(RNN['inputWN'])
                print(RNN['pVars'][-1])
        else:
            traj_list.append(low_rank_rnn(len(init_x),T,I,W,init_x=init_x+epsilon*np.random.randn()))
    if fit_rnn==True:
        return traj_list, full_traj, RNN_dict
    else:
        return traj_list, full_traj
    
    
def plot_vfield_net(V,res=10):
    V_c = np.vstack(V)
    x_min, y_min = np.min(V_c[:,0]), np.min(V_c[:,1])
    x_max, y_max = np.max(V_c[:,0]), np.max(V_c[:,1])
    
    x = np.linspace(x_min-x_min/10,x_max+x_max/10,res)
    y = np.linspace(y_min-y_min/10,y_max+y_max/10,res)
    X,Y = np.meshgrid(x,y)

    rho = np.histogram2d(V_c[:,0],V_c[:,1],bins=(x,y))
    D_rho = np.gradient(rho[0])
    
    plt.contourf(X[:-1,:-1].T,Y[:-1,:-1].T,-rho[0],cmap='afmhot')
    # plt.quiver(X[1:,1:].T,Y[1:,1:].T,-D_rho[1],-D_rho[0], scale=10,scale_units='xy')
    # plt.plot(V_c[:,0],V_c[:,1],'b')
    for i in range(len(V)):
        plt.plot(V[i][0,0],V[i][0,1],'*')
        plt.plot(V[i][:,0],V[i][:,1],alpha=0.05)
        plt.plot(V[i][-1,0],V[i][-1,1],'+')
    
def divergence_avg(V): 
    V_stk = np.stack(V)
    dist_through_time = np.zeros([np.shape(V_stk)[0],np.shape(V_stk)[0],np.shape(V_stk)[1]])
    for t in range(len(V_stk[0,:,0])):
        dist_through_time[:,:,t] = pairwise_distances(V_stk[:,t,:])
    return dist_through_time


def plot_circl_eig(r):
    rd = np.abs(r)
    r_perc = np.percentile(rd,95)
    theta = np.linspace(0,2*np.pi,100)
    plt.plot(1*np.cos(theta),1*np.sin(theta),'k')
    plt.plot(r_perc*np.cos(theta),r_perc*np.sin(theta),color='g')
    return r_perc
        

def complex_regression(y,fourier_order,fq_step=1):
    X = np.zeros([len(y),fourier_order],dtype='complex')
    for i in range(1,fourier_order):
        Fourier_f = np.exp(fq_step*i*1j*np.linspace(0,2*np.pi,len(y)))
        X[:,i] = Fourier_f
    b = np.linalg.pinv(X.conj().T@X)@X.conj().T@y
    return np.dot(X,b),b
   
def get_boundary(x):
    hull = ConvexHull(x)
    points = hull.points
    vertices = hull.vertices
    return np.squeeze(np.array([[points[vertices,0],points[vertices,1]]]))
    
def to_complex(z):
    return z[:,0]+1j*z[:,1]
    
def to_real(z):
    return np.array([np.real(z),np.imag(z)])

def Duffing_oscillator(mu,iters,gamma,delta,alpha,beta,omega,init_x=np.random.randn(2),plot_vfield=False):
    X = np.zeros([2,iters])
    X[:,0] = init_x
    # t = np.linspace(0,2*np.pi,iters)
    for i in range(iters-1):
        dy = -delta*X[1,i]-alpha*X[0,i]-beta*X[0,i]**3+gamma*np.cos(omega*mu*i)
        X[0,i+1] = X[0,i]+mu*X[1,i]
        X[1,i+1] = X[1,i]+mu*dy
    if plot_vfield==True:
        grid_spacing = np.linspace(-5,5,100)
        x,y = np.meshgrid(grid_spacing,grid_spacing)
        dxdt = y
        dydt = -delta*y-alpha*x-beta*x**3#+gamma*np.cos(omega*mu*i)
        plt.figure()
        plt.quiver(x,y,dxdt,dydt)
    return X
    
def plot_stimtime_funct(X,xplots,yplots):
    plt.figure(dpi=200)
    for x in range(1,xplots*yplots+1):
        plt.subplot(xplots,yplots,x)
        plt.imshow(X[x-1,:,:].T)
        plt.xticks([])
        plt.yticks([])
    
    
'''
    Regression with RBF kernels:
    x: a pointsXdim array for all the points at which the function is to be evaluated
    c: a list of centers of shape pointsXdim, where they repeat for each entry of points
'''
def RBF_kernels(x,c):
    X = np.zeros([x.shape[0]*x.shape[1],len(c)])
    for i in range(c):
        X[:,i] = np.exp(np.sqrt(np.sum((x-c)**2,0))).flatten()
    return X

def RBF_regress(y,x,centers):
    X_design = RBF_kernels(x,centers)
    beta = y @ np.pinv(X_design)
    return beta
    
    
    
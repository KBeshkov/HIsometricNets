from Algorithms import *

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


def EM_dist(X,Y,L1 = 0,L2 = 0,D = 0,norm='euclidean'):
    ''' Earth mover's distance as a linear assignment problem:
        
        Input:
            X: an array of inputs (eg spikes)
            Y: an array of inputs
            L1 = a list of cells that are being used in X
            L2 = a list of cells that are being used in Y
            D: a precomputed disance matrix
            
        Output: the distance between clusters
    '''    
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



def binary_hierarchical_labeling(nclasses,npoints): 
    labels = []
    for l in range(len(nclasses)):
        labels.append(np.zeros(npoints))
        for k in range(nclasses[l]):
            interval = int(npoints/nclasses[l])
            labels[l][k*interval:(k+1)*interval] = k
    return labels
        
        
        
def hierarchical_labeling_from_data(labels,pairings):
       lbls_new = []
       for l in labels:
           lbls_new.append(np.where(l == pairings)[0][0])
       return np.asarray(lbls_new)
 


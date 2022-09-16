#Topology of rigid transformation group
import numpy as np


a, b = np.linspace(-1,1,20), np.linspace(-1,1,20)
theta = np.linspace(0,4*np.pi,20)

def sample_rig_mats(angles,translations):
    A = []
    for t in angles:
        for a in translations[0]:
            for b in translations[1]:
                T = np.array([[0,0,a],
                              [0,0,b],
                              [0,0,1]])
                R = np.array([[np.cos(t), -np.sin(t), 0],
                              [np.sin(t),  np.cos(t), 0],
                              [        0,          0, 1]])
                A.append(T+R)
    return A


def pair_frob_dist(L):
    d = np.zeros([len(L),len(L)])
    for i in range(len(L)):
        for j in range(len(L)):
            if i > j:
                d[i,j] = np.sum((L[i]-L[j])**2)
    return d+d.T
    
    
A = sample_rig_mats(theta, [a,b])
D = pair_frob_dist(A)

H_A = tda(D,distance_matrix=True,maxdim=2,n_perm=500)['dgms']

plot_diagrams(H_A)

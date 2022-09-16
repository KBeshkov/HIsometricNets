import sys,os
sys.path.append(os.getcwd())
sys.path.append(os.getcwd()[:-7]+'Analysis')
from Algorithms import *
from metrics import *

#%% Define input manifolds
N_S1 = 200 #the number of points from the manifold which to sample
mfld_gen = Manifold_Generator()

S1 = mfld_gen.S1(N_S1,1)
line = np.linspace(0,1,N_S1)

#Construct neuron function with 90 degree rotation invariance
def delta_mfld(x,y):
    X = np.zeros([len(x),1])
    for i in range(len(x)):
        if np.linalg.norm(x[i,:]-y)==0:
            X[i] = 1
    return X
phi = lambda x,w: np.tanh(np.sum(w@x,1))

N_nrns = 100
incrs = np.linspace(1,2,int(N_S1/2))#[np.newaxis]
weights = (np.random.randn(N_S1,N_nrns)).T
for i in range(int(N_S1/2)):
    rvect = np.random.randn(N_nrns)#,1)
    weights[:,i] = incrs[i]*rvect#@incrs
    weights[:,int(N_S1/2)+i] = incrs[i]*rvect#@incrs

transformed_mfld = np.zeros([N_nrns,N_S1])
for i in range(N_S1):
    transformed_mfld[:,i] = phi(delta_mfld(S1.T,S1[:,i]).T,weights[:,i,np.newaxis])
    
mfld_pca = PCA(n_components=2).fit_transform(transformed_mfld.T)

plt.plot(mfld_pca[:,0],mfld_pca[:,1],'.')

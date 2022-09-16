import sys,os
sys.path.append(os.getcwd())
sys.path.append(os.getcwd()[:-7]+'Analysis')
from Algorithms import *
from metrics import *
import matplotlib.pyplot as plt
# plt.rcParams['font.family'] = 'Arial'

#%%
N = 1000
X,Y = np.meshgrid(np.linspace(-5,5,N),np.linspace(-5,5,N))
xy = np.vstack([X.flatten(),Y.flatten()])

layers = 8
dims = 16*np.ones(layers).astype(int)#np.logspace(1,layers,layers,base=2).astype(int)
dims[0] = 2
nn_maps = []
weights = ortho_group.rvs(dims[0])
for l in range(layers-1):
    biases = 1*np.random.randn(dims[l+1])
    if l == 0:
        weights_in = 1*np.random.randn(dims[l+1],dims[l])
        nn_maps.append(np.tanh(weights_in@xy+np.diag(biases)@np.ones([dims[l+1],N**2])))
    else:
        weights = 1*np.random.randn(dims[l+1],dims[l])#ortho_group.rvs(dims[l])#
        nn_maps.append(np.tanh(weights@nn_maps[-1]+np.diag(biases)@np.ones([dims[l+1],N**2])))

#%%plots
import matplotlib.pyplot as plt

plot_grid(nn_maps,fft=False)

import sys,os
sys.path.append(os.getcwd())
sys.path.append(os.getcwd()[:-7]+'Analysis')
from Algorithms import *
from metrics import *
from ML_models import *
plt.style.use('default')#../misc/report.mplstyle')
#%%Line to circle network
N=20
mfld_gen = Manifold_Generator()

x,y = np.meshgrid(np.linspace(0,2,N),np.linspace(0,2,N))
R1 = torch.tensor(np.vstack([x.flatten(),y.flatten()])).T#torch.tensor(mfld_gen.S2(N, 0.5)).T#torch.linspace(0,2,N).reshape(-1,1)
S1 = torch.tensor(mfld_gen.S2(N,1,0.5)).T#mfld_gen.S1(N, 0.5)).T+2

nlayers = 2
n_nrns = 40
network_params = {'n_neurons': n_nrns, 'n_inputs': 2,'n_classes': 1,
                  'projection': [],'weights': [],'n_layers': nlayers}
HRIM = HierarchicalRepresentationNetwork(NN_class(**network_params).double(),
                                         dmat=torch.tensor(pairwise_distances(S1)),savepoints=100,learning_rate=5e-4)

labels = binary_hierarchical_labeling([1], N**2)
costs = [[nn.CrossEntropyLoss(),MetricPreservationLoss(10, 0).Loss] for i in range(nlayers)]

HRIM_data = HRIM.train(R1, labels, costs, epochs=1000)

reducer = PCA(n_components=3)#umap.UMAP(n_neighbors=15,min_dist=0.75,n_components=3)#
embeddings = [reducer.fit_transform(dat[0][0].detach().numpy())
              for dat in HRIM_data[0]]
plt.figure()
plt.imshow(pairwise_distances(S1)-pairwise_distances(HRIM_data[0][-1][0][0].detach().numpy()))
plt.colorbar()
#%%Animations
def animate(i,embedding):
    ax.axis('off')
    ax.cla()
    ax.scatter(embedding[i][:,0],embedding[i][:,1],embedding[i][:,2],s=8,alpha=0.4)
    ax.set_xlim(-2,2)
    ax.set_ylim(-2,2)
    ax.set_zlim(-2,2)

#%%Circle animation
fig = plt.figure(figsize=(6,6),dpi=200,constrained_layout=True)
ax = plt.subplot(111,projection='3d')
anim = animation.FuncAnimation(fig, animate, blit=False, frames = 60,fargs=[embeddings],interval=500)
# anim.save('../Figures/sphere_to_torus.gif', writer='imagemagick', fps=60)
    
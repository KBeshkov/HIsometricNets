#Hierarchical Representations through locally isomorphic embeddings
import sys,os
sys.path.append(os.getcwd())
sys.path.append(os.getcwd()[:-7]+'Analysis')
from Algorithms import *
from metrics import *
from ML_models import *
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
plt.rcParams['font.family'] = 'Arial'
pallette = plt.rcParams['axes.prop_cycle'].by_key()['color']
pallette2 = sns.color_palette('colorblind')
plt.style.use('default')#../misc/report.mplstyle')

#%%Topology of gratings
N_angles = 30
N_sfqs = 30
res = 40
angles = np.linspace(0,np.pi,N_angles)
sfqs = np.linspace(1,16,N_sfqs)
stimuli = np.zeros([N_sfqs, N_angles,  res,res])

for i in range(N_sfqs):
    for j in range(N_angles):
        stimuli[i,j] = generate_gratings(res, angles[j], sfqs[i])#generate_gabors(res,angles[j],sfqs[i],3,1)#
    
stimuli_tovec = stimuli.reshape(N_angles*N_sfqs,res**2)

reducer = umap.UMAP(n_neighbors=50,min_dist=1,n_components=3)#PCA(n_components=3)
stimuli_mfld = reducer.fit_transform(stimuli_tovec).T

fig = plt.figure(dpi=200)
ax1 = fig.add_subplot(111,projection='3d')
ax1.scatter(stimuli_mfld[0],stimuli_mfld[1],stimuli_mfld[2],
            c=np.arange(0,N_angles*N_sfqs),
            cmap=sns.color_palette("flare", as_cmap=True),
            alpha=0.2)
ax1.axis('off')
# plt.savefig('../Figures/gratings_mfld.png',transparent=True,dpi=1000)


for i in range(1,5):
    for j in range(1,5):
        plt.figure()
        plt.imshow(stimuli[i*12,j*12],cmap='gray')
        plt.axis('off')
        # plt.savefig('../Figures/'+str(i*12)+'_'+str(j)+'_grating.png',transparent=True)
    
Phom = Persistent_Homology()
stimuli_hom = Phom(stimuli_tovec,geodesic,False,1,100)
barcode_plot(stimuli_hom[1],dims=2)


#%% Training a network to generalize
split_int = int(N_angles*N_sfqs/2)
labels = torch.zeros(2*split_int)
count = 0
for i in range(N_sfqs):
    for j in range(N_angles):
        if angles[j]>np.pi/2:
            labels[count] = 1
        count += 1
train_labels = labels[:split_int]
test_labels = labels[split_int:]
labels = binary_hierarchical_labeling([2,4,8,5,3],N_angles*N_sfqs)
data = torch.tensor(stimuli_tovec)

D = len(stimuli_tovec)
N_nrns = 50
n_layers = 5
costs = [[nn.CrossEntropyLoss(),MetricPreservationLoss(100, 0).Loss] for i in range(n_layers)]
network_params = {'n_neurons': N_nrns, 'n_inputs': res**2,'n_classes': 8,
                  'projection': [],'weights': [],'n_layers': n_layers}
HRIM = HierarchicalRepresentationNetwork(NN_class(**network_params).double())
HRIM_data = HRIM.train(data, labels, costs, epochs=2000)

#%%
Mfld_test = data + 0.001*torch.randn(data.shape)
Mfld_transform = reducer.fit_transform(Mfld_test)
fig = plt.figure(dpi=200)
for i in range(n_layers):
    pred = HRIM.prop_forward(Mfld_test, i+1)[0]
    mfld_pred = reducer.fit_transform(pred.detach().numpy()).T
    ax1 = fig.add_subplot(2,n_layers,i+1,projection='3d')
    ax2 = fig.add_subplot(2,n_layers,i+1+n_layers,projection='3d')
    for j in np.unique(labels[i]).astype(int):
        ax1.scatter(Mfld_transform[labels[i]==j,0],Mfld_transform[labels[i]==j,1],Mfld_transform[labels[i]==j,2],s=2,alpha=0.4)
        ax2.scatter(mfld_pred[0,labels[i]==j],mfld_pred[1,labels[i]==j],mfld_pred[2,labels[i]==j],s=2,alpha=0.4)
        ax1.axis('off') 
        ax1.grid('off')
        ax2.axis('off') 
        ax2.grid('off')
    plt.tight_layout()
#%%
pca_pred = umap.UMAP(n_neighbors=50,min_dist=1,n_components=3).fit_transform(predictions[0]).T#PCA(n_components=3).fit_transform(predictions[1]).T#

fig=plt.figure(dpi=300,figsize=(2,2))
ax2 = fig.add_subplot(1,1,1,projection='3d')
for i in range(len(np.unique(test_labels))):
    ax2.scatter(pca_pred[0,test_labels==i],pca_pred[1,test_labels==i],pca_pred[2,test_labels==i],
                s=.5,alpha=0.7,color=pallette2[i])
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_zticks([])
    
plt.tight_layout()

#%%Critical point space

Theta, Q = np.meshgrid(angles,sfqs)
h = np.cos(Q*np.cos(Theta))
# dh = np.sin(Q*np.cos(Theta))*(np.sin(Theta)*Q+np.cos(Theta))
grad_h = [np.sin(Q*np.cos(Theta))*np.sin(Theta)*Q, -np.sin(Q*np.cos(Theta))*np.cos(Theta)]

plt.figure()
plt.contourf(Q,Theta,h,cmap='RdBu_r')
plt.quiver(Q,Theta,grad_h[0],grad_h[1],width=np.arange(2e-3,1,Q.size),cmap='RdBu_r')
plt.colorbar()
# plt.contour(Q,Theta,dh,2,cmap='gray')
plt.title('$df($\theta$,q)$')

#%%Hessian
H = lambda q,angl: np.array([[-(q**2)*np.cos(q*np.cos(angl))*np.sin(angl)**2+q*np.sin(q*np.cos(angl))*np.cos(angl),
                              np.sin(q*np.cos(angl))*np.sin(angl)+q*np.cos(q*np.cos(angl))*np.cos(angl*np.sin(angl))],
                             [np.sin(q*np.cos(angl))*np.sin(angl)+q*np.cos(q*np.cos(angl))*np.cos(angl*np.sin(angl)),
                              -np.cos(q*np.cos(angl))*np.cos(angl)**2]])
Hessian = H(Q,Theta)
plt.figure()
plt.subplot(2,2,1)
plt.contourf(Q,Theta,Hessian[0,0],cmap='RdBu_r',vmin=-1,vmax=1)
plt.subplot(222)
plt.contourf(Q,Theta,Hessian[0,1],cmap='RdBu_r',vmin=-1,vmax=1)
plt.subplot(223)
plt.contourf(Q,Theta,Hessian[1,0],cmap='RdBu_r',vmin=-1,vmax=1)
plt.subplot(224)
plt.contourf(Q,Theta,Hessian[1,1],cmap='RdBu_r',vmin=-1,vmax=1)

#eigenvalues on critical manifolds
crit_mfld = np.meshgrid(sfqs,np.arccos(np.pi/sfqs))#np.vstack([sfqs,np.arccos(2*np.pi/sfqs)])#
crit_mfld[0][np.isnan(crit_mfld)[0]] = 0
crit_mfld[1][np.isnan(crit_mfld)[1]] = 0
Hess_crit = H(crit_mfld[0],crit_mfld[1])
Hess_crit[np.isnan(Hess_crit)]=0
Hess_eig = [np.zeros([N_sfqs,N_angles]),np.zeros([N_sfqs,N_angles])]
for i in range(Hessian.shape[-1]):
    for j in range(Hessian.shape[-1]):
        eig = np.linalg.eig(Hess_crit[:,:,i,j])[0]
        Hess_eig[0][i,j] = eig[0]
        Hess_eig[1][i,j] = eig[1]

        
plt.figure()
plt.subplot(211)
plt.contourf(crit_mfld[0],crit_mfld[1],Hess_eig[0],cmap='RdBu_r',vmin=-1,vmax=1)
plt.subplot(212)
plt.contourf(crit_mfld[0],crit_mfld[1],Hess_eig[1],cmap='RdBu_r',vmin=-1,vmax=1)

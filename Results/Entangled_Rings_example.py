#Entangled loops example
import sys,os
sys.path.append(os.getcwd())
sys.path.append(os.getcwd()[:-7]+'Analysis')
from Algorithms import *
from metrics import *
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import pickle

plt.rcParams['font.family'] = 'Arial'
plt.rcParams.update({'font.size': 14})
pallette = plt.rcParams['axes.prop_cycle'].by_key()['color']
pallette2 = sns.color_palette('colorblind')
plt.style.use('default')#../misc/report.mplstyle')
#%% Load entangled loop data
with open('/Users/constb/Repos/copies/StimMfldRnn/Data/train_rings.pkl', 'rb') as handle:
    Toy_data_train = pickle.load(handle)
with open('/Users/constb/Repos/copies/StimMfldRnn/Data/test_rings.pkl', 'rb') as handle:
    Toy_data_test = pickle.load(handle)
    
#%%ML
from ML_models import *

N_net = 100
n_layers = 4
data = Toy_data_train[0].T

D = len(data)
dat_test = Toy_data_test[0].T
labels = Toy_data_train[1]
labels_test = Toy_data_test[1]
nclass = len(np.unique(labels))

train_data = [torch.tensor(data),torch.tensor(labels)]
test_data = [torch.tensor(dat_test),torch.tensor(labels_test)]
performance = []

#%%
model = NN_class(N_net,D,len(np.unique(labels)),[],[],n_layers).double()
dmat = torch.cdist(train_data[0].T,train_data[0].T)
init_run = torch.tensor(predict(model,train_data[0].T)[1])
criterion = MetricPreservationLoss(lambda1=100,lambda2=0)
optimizer = optim.Adam(model.parameters())
epoch_n = 500

train_type = 'Metric'
if train_type=='Metric':
    loss_weights = [1,100]
elif train_type =='CSE':
    loss_weights = [1,0]
    
train_params = {'net':model,'train_dat':train_data,'optimizer':optimizer,'criterion_weights':loss_weights,
                'epochs':epoch_n,'batch_sz':len(train_data[0].T),'dmat':dmat}


loss_curve = train(**train_params)

plt.plot(loss_curve[-1])
plt.ylabel('$\\frac{||D_{\Phi}||_\infty}{||D||_\infty}$')

plt.figure()
for count,i in enumerate(loss_curve[:-2]): plt.plot(i,linewidth=2,color=pallette2[count])
plt.legend(['Total','Cross-Entropy','Metric'])
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.grid('on')
plt.xscale('log')
plt.yscale('log')
# plt.savefig('../Figures/Loss'+train_type+'Ring.png', transparent=True,dpi=1000)
#%%
predictions = predict(model,test_data[0].T)
predicted_labels = np.argmax(predictions[1],1)
performance = np.sum(labels_test==predicted_labels)/len(labels_test)
print(performance)

#%%
fig = plt.figure(dpi=300,figsize=(2,2))
        
ax1 = fig.add_subplot(1,1,1,projection='3d')               
for i in range(nclass):
    ax1.scatter(dat_test[0,labels_test==i],dat_test[1,labels_test==i],dat_test[2,labels_test==i],
                s=.5,alpha=0.7,color=pallette2[i])
ax1.set_axis_off()
# plt.savefig('../Figures/Rings_class.png',transparent=True,dpi=1000)

pca_pred = umap.UMAP(n_neighbors=40,min_dist=1,n_components=3).fit_transform(predictions[0]).T

fig=plt.figure(dpi=300,figsize=(2,2))
ax2 = fig.add_subplot(1,1,1,projection='3d')
for i in range(nclass):
    ax2.scatter(pca_pred[0,labels_test==i],pca_pred[1,labels_test==i],pca_pred[2,labels_test==i],
                s=.5,alpha=0.7,color=pallette2[i])
ax2.set_axis_off()
plt.tight_layout()
# plt.savefig('../Figures/Rings_class_dec_'+train_type+'.png',transparent=True,dpi=1000)

#%%Computing the Betti numbers through layers
Phomer = Persistent_Homology()
phoms_full = [Phomer(model.lim_forward(test_data[0].T,i).detach().numpy(),geodesic,True,1,100) for i in range(n_layers)]
phoms_1 = [Phomer(model.lim_forward(test_data[0][:,labels_test==0].T,i).detach().numpy(),geodesic,True,1,100) for i in range(n_layers)]
phoms_2 = [Phomer(model.lim_forward(test_data[0][:,labels_test==1].T,i).detach().numpy(),geodesic,True,1,100) for i in range(n_layers)]


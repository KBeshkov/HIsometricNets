# MNIST classification using topologically regularized FF network
import sys,os
sys.path.append(os.getcwd())
sys.path.append(os.getcwd()[:-7]+'Analysis')
from Algorithms import *
from metrics import *
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
from torch.utils.data import SubsetRandomSampler
import seaborn as sns
pallette2 = sns.color_palette('dark')
plt.style.use('default')#../misc/report.mplstyle')

#%%
Data_train = datasets.MNIST('../Data/',transform=Compose([ToTensor(), Lambda(lambda x: torch.flatten(x))]))
Data_test = datasets.MNIST('../Data/',train=False,transform=Compose([ToTensor(), Lambda(lambda x: torch.flatten(x))]))

train_inds = torch.randperm(len(Data_train))[:2500]


train_data = torch.vstack([Data_train[i][0] for i in train_inds]).T
test_data = torch.vstack([Data_test[i][0] for i in range(len(Data_test))]).T
train_labels = torch.tensor([Data_train[i][1] for i in train_inds])
test_labels = torch.tensor([Data_test[i][1] for i in range(len(Data_test))])

train_labels_hierarch1 = hierarchical_labeling_from_data(train_labels,torch.tensor([[0,1,2,3,4],[5,6,7,8,9]]))
train_labels_hierarch2 = hierarchical_labeling_from_data(train_labels,torch.tensor([[0,1],[2,3],[4,5],[6,7],[8,9]]))
labels = [train_labels_hierarch1,train_labels_hierarch2,train_labels.detach().numpy()]
test_labels_hierarch1 = hierarchical_labeling_from_data(test_labels,torch.tensor([[0,1,2,3,4],[5,6,7,8,9]]))
test_labels_hierarch2 = hierarchical_labeling_from_data(test_labels,torch.tensor([[0,1],[2,3],[4,5],[6,7],[8,9]]))
labels_test = [test_labels_hierarch1,test_labels_hierarch2, test_labels.detach().numpy()]

train_data = [train_data.double(), train_labels]
test_data = [test_data.double(), test_labels]
#%%ML
from ML_models import *


N_net = 200
n_layers = 3
D = 28**2
n_classes = len(Data_train.classes)
network_params = {'n_neurons': N_net, 'n_inputs': D,'n_classes': n_classes,
                  'projection': [],'weights': [],'n_layers': n_layers}
model = HierarchicalRepresentationNetwork(NN_class(**network_params).double(),savepoints=5)
# model = NN_class(N_net,D,n_classes,[],[],n_layers).double()

dmat = torch.cdist(train_data[0].T,train_data[0].T)
#%%Training
# criterion1 = nn.CrossEntropyLoss()
# criterion2 = TopologicalDestroyerLoss(model,1,1,
#                                       torch.tensor(bound_S2), 
#                                       torch.tensor(repel_S2))
# criterion2 = ManifoldLagrangeLoss(1,'Sn')
# criterion = MetricPreservationLoss(1,0)
# optimizer = optim.Adam(model.parameters())

# loss_curve = train(model,train_data,optimizer,[criterion.Loss],2000,len(dmat),'MetricSave',dmat,20)#'assigned',aspoint)

# plt.figure()
# for count,i in enumerate(loss_curve[:-1]): plt.plot(i,linewidth=2,color=pallette2[count])
# plt.legend(['Total','Cross-Entropy','Metric'])
# plt.xlabel('iterations')
# plt.ylabel('Loss')
# plt.grid('on')
# plt.xscale('log')
# plt.yscale('log')
# plt.savefig('../Figures/LossMNIST_Metric.png', transparent=True,dpi=1000)

costs = [[nn.CrossEntropyLoss(),MetricPreservationLoss(10, 0).Loss] for i in range(n_layers)]

HRIM_data = model.train(train_data[0].T, labels, costs, epochs=200)

#%%
predictions = model.prop_forward(test_data[0].T, 4)[1]
predicted_labels = np.argmax(predictions.detach().numpy(),1)
performance = np.sum(test_labels.detach().numpy()==predicted_labels)/len(test_labels)
print(performance)

#%%
dimred_test = umap.UMAP(n_neighbors=10,min_dist=0.75,n_components=2).fit_transform(test_data[0].T).T
pca_pred = umap.UMAP(n_neighbors=10,min_dist=0.75,n_components=2).fit_transform(predictions[0]).T

#%%
fig = plt.figure(dpi=300,figsize=(5,2))

# gs = GridSpec(4,20,fig)
# count = 0
# for i in range(int((n_classes)/2)):
#     for j in range(int((n_classes)/2)):
#         ax1 = fig.add_subplot(gs[2*i:2*(i+1),2*j:2*(j+1)], projection='3d') 
#         ax1.scatter(dimred_test[0,test_labels==i],dimred_test[1,test_labels==i],dimred_test[2,test_labels==i],
#                     s=.5,alpha=0.5,color=pallette2[count])
#         ax1.set_xticks([])
#         ax1.set_yticks([])
#         ax1.set_zticks([])
#         count += 1
        
ax1 = fig.add_subplot(1,2,1)#,projection='3d')               
for i in range(n_classes-1):
    ax1.scatter(dimred_test[0,test_labels==i],dimred_test[1,test_labels==i],s=.5,alpha=0.3,color=pallette2[i])#dimred_test[2,test_labels==i],
             #   s=.5,alpha=0.3,color=pallette2[i])
    ax1.set_xticks([])
    ax1.set_yticks([])
    # ax1.set_zticks([])


# count = 0 
# for i in range(int((n_classes)/2)):
#     for j in range(int((n_classes)/2)):
#         ax2 = fig.add_subplot(gs[2*i:2*(i+1),4+2*j:4+2*(j+1)], projection='3d') 
#         ax2.scatter(pca_pred[test_labels==count,0],pca_pred[test_labels==count,1],pca_pred[test_labels==count,2],
#                     s=.5,alpha=0.5,color=pallette2[count]) 
#         # ax2.scatter(predictions[1][labels_test==count,0],predictions[1][labels_test==count,1],predictions[1][labels_test==count,2],
#                     # s=.5,alpha=0.5,c=pallette[count]) 
#         ax2.set_xticks([])
#         ax2.set_yticks([])
#         ax2.set_zticks([])
#         count += 1

ax2 = fig.add_subplot(1,2,2)#,projection='3d')
for i in range(n_classes):
    ax2.scatter(pca_pred[0,test_labels==i],pca_pred[1,test_labels==i],s=.5,alpha=0.3,color=pallette2[i])#,pca_pred[2,test_labels==i],
                # s=.5,alpha=0.3,color=pallette2[i])
    ax2.set_xticks([])
    ax2.set_yticks([])
    # ax2.set_zticks([])
    
plt.tight_layout()
# plt.savefig('../Figures/classMNIST_CSE_Metric_sep.png',transparent=True,dpi=1000)

#%%
Phom = Persistent_Homology()
pdiags_in = []
pdiags_out = []
for i in range(n_classes):
    pdiags_in.append(Phom(test_data[0].T[test_labels==i],pairwise_distances,False,2,250)[1])#tda(test_data[0].T[labels_test==i],maxdim=2)['dgms'])
    pdiags_out.append(Phom(predictions[0][test_labels==i],pairwise_distances,False,2,250)[1])
    barcode_plot(pdiags_in[-1],dims=3)
    barcode_plot(pdiags_out[-1],dims=3)
    # plt.savefig('../Figures/pers_hom_classMNIST_Metric_'+str(i)+'.png',transparent=True,dpi=200)
    







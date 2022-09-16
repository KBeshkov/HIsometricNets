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
# plt.rcParams['font.family'] = 'Arial'
pallette = plt.rcParams['axes.prop_cycle'].by_key()['color']
pallette2 = sns.color_palette('tab20')
plt.style.use('default')#../misc/report.mplstyle')

#%%
nlayers = 4
n_nrns = 100

N = 16
partition = [[[0,1,2,3],[4,5,6,7]],[[0,1],[2,3],[4,5],[6,7]],
          [[0],[1],[2],[3],[4],[5],[6],[7]]]
labels = binary_hierarchical_labeling([2,4,8], N**2)  
          
noise = 1e-3
Mfld = torch.tensor(Manifold_Generator().T2(N, 1, 0.66)).T[:,:,None]
Mfld_train = torch.utils.data.TensorDataset(Mfld + noise*torch.randn(Mfld.size()),torch.tensor(labels[-1]))
Mfld_train = MfldDataset(Mfld_train)
Mfld_test = torch.utils.data.TensorDataset(Mfld + noise*torch.randn(Mfld.size()),torch.tensor(labels[-1]))
Mfld_test = MfldDataset(Mfld_test)


nclass = len(labels)
data_train_hierarchy = HierarchicalDataset(Mfld_train,scale_img=False)
data_test_hierarchy = HierarchicalDataset(Mfld_test,scale_img=False)


data_part_train = data_train_hierarchy.__hierarchy__(partition)
data_part_test = data_test_hierarchy.__hierarchy__(partition)

hierarchical_trainloader = torch.utils.data.DataLoader(data_train_hierarchy, batch_size=N**2, shuffle=True)
hierarchical_testloader = torch.utils.data.DataLoader(data_test_hierarchy, batch_size=N**2, shuffle=False)


Iso_coefs = np.logspace(-1,7,8)
gammas = np.logspace(-1,2,8)
epsilon = 1e-2
robustness = np.zeros([len(gammas),len(Iso_coefs)])
performance = np.zeros([len(gammas),len(Iso_coefs)])
network_params = {'n_neurons': n_nrns, 'n_inputs': 3,'n_classes': 8,
                  'projection': [],'weights': [],'n_layers': nlayers*nclass}

for count1,g in enumerate(gammas):
    for count2,ic in enumerate(Iso_coefs):
    
        HRIM = HierarchicalRepresentationNetwork(NN_class(**network_params).double(),savepoints=5)
        
        costs = [[nn.CrossEntropyLoss(),MetricPreservationLoss(ic, g).Loss],
                 [nn.CrossEntropyLoss(),MetricPreservationLoss(ic, g).Loss],
                 [nn.CrossEntropyLoss(),MetricPreservationLoss(ic, g).Loss]]
        
        HRIM_data = HRIM.train(hierarchical_trainloader, costs, epochs=1000,plot=False)
        
        #%%
        HRIM_test = HRIM.test(hierarchical_testloader)
        performance[count1,count2] = HRIM_test[-1]    
        # #%%
        # reducer = umap.UMAP(n_neighbors=45,min_dist=0.75,n_components=3)
        # fig = plt.figure(dpi=200)
        # for i in range(nclass):
        #     pred = HRIM.prop_forward(Mfld_test.data[:,:,0], i+1)[0]
        #     mfld_pred = reducer.fit_transform(pred.detach().numpy()).T
        #     ax1 = fig.add_subplot(2,nclass,i+1,projection='3d')
        #     ax2 = fig.add_subplot(2,nclass,i+1+nclass,projection='3d')
        #     for j in np.unique(labels[i]).astype(int):
        #         ax1.scatter(Mfld[labels[i]==j,0],Mfld[labels[i]==j,1],Mfld[labels[i]==j,2],s=2,alpha=0.4)
        #         ax2.scatter(mfld_pred[0,labels[i]==j],mfld_pred[1,labels[i]==j],mfld_pred[2,labels[i]==j],s=2,alpha=0.4)
        #         ax1.axis('off') 
        #         ax1.grid('off')
        #         ax2.axis('off') 
        #         ax2.grid('off')
        #     plt.tight_layout()
        #     # plt.savefig('../Figures/class_dec_Metric_layer_'+str(i+1)+'.png',transparent=True,dpi=1000)
    
    
    #%%addverserial attacks
    
        attack_type = 'fgsm'
        adv_attacks = []
        robustness[count1,count2] = HRIM.hierarchical_test_attack(hierarchical_testloader,costs,
                                                         epsilon=epsilon, attack_method=attack_type)[0][2]
        print((count1,count2))
    
# #%%Animation of mfld separation between layers and through training
# reducer = umap.UMAP(n_neighbors=15,min_dist=0.75,n_components=3)#PCA(n_components=3)
# HRIM_reps = []
# for i in range(nlayers):
#     HRIM_reps.append([reducer.fit_transform(mfld[i][0].detach().numpy()) for mfld in HRIM_data[0]])

# #%%    
# def animate(i,embedding,labels):
#     for j in range(len(embedding)):
#         ax_ = plt.subplot(1,len(embedding),j+1,projection='3d')
#         ax_.cla()
#         ax_.scatter(embedding[j][i][:,0],embedding[j][i][:,1],embedding[j][i][:,2],s=6,alpha=0.4,c=labels[j],cmap='tab20')
#         ax_.axis('off') 
#         ax_.grid('off')
#         ax_.set_title('Layer '+str(j+1))
        
# fig = plt.figure(figsize=(6,2),dpi=200,constrained_layout=True)
# anim = animation.FuncAnimation(fig, animate, blit=False, frames = 60,fargs=(HRIM_reps,labels),interval=1000)
# # anim.save('../Figures/class_HRIM_CSE.gif', writer='imagemagick', fps=60)





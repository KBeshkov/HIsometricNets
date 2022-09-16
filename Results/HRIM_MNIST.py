#%% MNIST HRIM
try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass
import sys,os
sys.path.append(os.getcwd())
sys.path.append(os.getcwd()[:-7]+'Analysis')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
import seaborn as sns
from torchvision import datasets, transforms
plt.style.use('default')#../misc/report.mplstyle')
plt.rcParams['font.family'] = 'Arial'
pallette2 = sns.color_palette('colorblind')
cust_cmap =sns.color_palette("rocket", as_cmap=True)
sys.path.append(os.getcwd())
sys.path.append(os.getcwd()[:-7]+'Analysis')
from Algorithms import *
from metrics import *
from ML_models import *

Iso_coef = 100

transform=transforms.Compose([transforms.ToTensor()])

data_train = datasets.MNIST('/Users/constb/Data/MNIST/', train=True, transform=transform)#, download=True)
data_test = datasets.MNIST('/Users/constb/Data/MNIST/', train=False, transform=transform)#,download=True)

data_train_hierarchy = HierarchicalDataset(data_train)
data_test_hierarchy = HierarchicalDataset(data_test)

partition = [[[0,1,2,3,4],[5,6,7,8,9]],
             [[0,1],[2,3],[4,5],[6,7],[8,9]],
             [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]]]

data_part_train = data_train_hierarchy.__hierarchy__(partition)
data_part_test = data_test_hierarchy.__hierarchy__(partition)


hierarchical_trainloader = torch.utils.data.DataLoader(data_train_hierarchy, batch_size=100, shuffle=True)
hierarchical_testloader = torch.utils.data.DataLoader(data_test_hierarchy, batch_size=100, shuffle=True)
  
precomputed_dmat=True
if precomputed_dmat == True:
    MNIST_dmat = torch.load(os.getcwd()[:-8]+'/Data/MNIST_dmat.pt').double() #Precomputed distance matrix between MNIST images
else:
    MNIST_dmat='standard'
res = data_train[0][0].size()[-1]

#%% Train Model
nlayers = 4
nclass = len(partition)
n_nrns = 100
network_params = {'n_neurons': n_nrns, 'n_inputs': res**2,'n_classes': 10,
                  'projection': [],'weights': [],'n_layers': nclass*nlayers}
HRIM = HierarchicalRepresentationNetwork(NN_class(**network_params).double().to('cpu'),savepoints=25)

costs = [[nn.CrossEntropyLoss(),MetricPreservationLoss(Iso_coef, 0.1).Loss],
         [nn.CrossEntropyLoss(),MetricPreservationLoss(Iso_coef, 0.1).Loss],
         [nn.CrossEntropyLoss(),MetricPreservationLoss(Iso_coef, 0.1).Loss]]

HRIM_data = HRIM.train(hierarchical_trainloader, costs, dmat=MNIST_dmat,epochs=5)

plt.figure()
plt.plot(HRIM_data[-1])
plt.ylabel('$\\frac{||D_{\Phi}||_\infty}{||D||_\infty}$')

#%% Test model
HRIM_test = HRIM.test(hierarchical_testloader)
print(HRIM_test)

#%% Manifold and cross-section plots
point_cloud = torch.reshape(data_test.data,[len(data_test),28**2])/255
#Use PCA to get the eigenvector which maximize the variance
pca_method = PCA()
pca_method.fit(point_cloud)
eigenvects = pca_method.components_[:2]

targets = data_test.targets
model_out = HRIM.prop_forward(point_cloud,80)
fx = model_out[0].detach().numpy()
pred = torch.argmax(model_out[1],1)

reducer = umap.UMAP(n_neighbors=50,min_dist=1,n_components=2)#PCA(n_components=3)
stimuli_mfld = reducer.fit_transform(fx).T

fig = plt.figure(dpi=200)
ax1 = fig.add_subplot(111)
ax1.scatter(stimuli_mfld[0],stimuli_mfld[1],
            c=targets,
            cmap=sns.color_palette("rocket_r",as_cmap=True),
            alpha=0.2)
ax1.axis('off')
# fig.savefig('../Figures/MNIST_HRIM_manifold_'+str(Iso_coef)+'.png',dpi=1000)

#%%cross-sections
nperts = 5000

prototype = point_cloud[4751]
projected_xy = (prototype@eigenvects.T).T
perturbed_prototypes = torch.zeros([nperts,len(prototype)])
projected_prototypes = torch.zeros([nperts,2])
epsilon = 10
for i in range(nperts):
    c1, c2 = (2*torch.rand(1)-1), (2*torch.rand(1)-1)
    perturbed_prototypes[i] = prototype + epsilon*c1*eigenvects[0] + epsilon*c2*eigenvects[1]
    # perturbed_prototypes[i] = torch.clamp(perturbed_prototypes[i],0,1)
projected_prototypes = (perturbed_prototypes@eigenvects.T).T

#propagate cross-sections through model
model_cross_out = HRIM.prop_forward(perturbed_prototypes,-1)
model_cross = model_cross_out[0]
model_cross_preds = torch.argmax(model_cross_out[1],1)

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.tricontourf(projected_prototypes[0],projected_prototypes[1],
            model_cross_preds,cmap=sns.color_palette("cubehelix_r",as_cmap=True))
ax.add_patch(
    patches.Rectangle(
        xy=(projected_xy[0]-.5, projected_xy[1]-.5),
        width=1, height=1, linewidth=1,
        color='black', fill=False))
ax.plot(projected_xy[0], projected_xy[1],'k*')

# plt.savefig('../Figures/HRIM_cross_section_'+str(Iso_coef)+'.png',dpi=1000)

#%% Distance difference function plot
metric = lambda x,y: torch.cdist(torch.unsqueeze(HRIM.prop_forward(x,-1)[0],0),HRIM.prop_forward(y,-1)[0])
DD = metric(prototype,perturbed_prototypes).squeeze()
prot_dmat = torch.cdist(torch.unsqueeze(prototype,0),perturbed_prototypes)

fig, ax = plt.subplots()
ax.set_aspect('equal')
im=ax.tricontourf(projected_prototypes[0],projected_prototypes[1],
            torch.abs(prot_dmat-DD)[0].detach().numpy(),cmap=sns.color_palette("rocket_r",as_cmap=True))
fig.colorbar(im)

# plt.savefig('../Figures/HRIM_dfield_cross_section_'+str(Iso_coef)+'.png',dpi=1000)

#%% Testing adverserial attacks

#define epsilon ranges
epsilons = np.logspace(-2,0,10)

attack_type = 'fgsm'
adv_attacks = []
for e in epsilons:
    adv_attacks.append(HRIM.hierarchical_test_attack(hierarchical_testloader,costs,
                                                     epsilon=e, attack_method=attack_type))

class_performance = np.hstack([adv_attacks[i][0] for i in range(len(epsilons))]).T
orig_images1 = [np.reshape(adv_attacks[i][1][2][2][5],[28,28]) for i in range(len(epsilons))]
orig_labels1 = [adv_attacks[i][1][2][0][5] for i in range(len(epsilons))]
orig_images2 = [np.reshape(adv_attacks[i][1][2][2][3],[28,28]) for i in range(len(epsilons))]
orig_labels2 = [adv_attacks[i][1][2][0][3] for i in range(len(epsilons))]
pert_images1 = [np.reshape(adv_attacks[i][1][2][3][5],[28,28]) for i in range(len(epsilons))]
pert_labels1 = [adv_attacks[i][1][2][1][5] for i in range(len(epsilons))]
pert_images2 = [np.reshape(adv_attacks[i][1][2][3][3],[28,28]) for i in range(len(epsilons))]
pert_labels2 = [adv_attacks[i][1][2][1][3] for i in range(len(epsilons))]

#%% Degree of error measured in terms of the distance between the MNIST predicted labels
orig_labels = [adv_attacks[i][1][2][0] for i in range(len(epsilons))]
pert_labels = [adv_attacks[i][1][2][1] for i in range(len(epsilons))]
distances = [torch.sum(torch.abs(orig_labels[i]-pert_labels[i])).detach().numpy() for i in range(len(epsilons))]
tree_distance_mat = tree_distances(partition)[1][len(partition[0])+len(partition[1]):,len(partition[0])+len(partition[1]):]
tree_distance = [np.sum(tree_distance_mat[orig_labels[i],pert_labels[i]]) for i in range(len(epsilons))]

# np.save('../Data/class_curves_'+attack_type+str(Iso_coef)+'.npy',class_performance[:,2])
# np.save('../Data/distances_'+attack_type+str(Iso_coef)+'.npy',np.array(tree_distance))
#%% Plots
class_names = [['0-4','5-9'],[['0-1'],['2-3'],['4-5'],['6-7'],['8-9']],
               [['0'],['1'],['2'],['3'],['4'],['5'],['6'],['7'],['8'],['9']]]


fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(nrows=8, ncols=8)

ax01 = fig.add_subplot(gs[:2,:4])
for i in range(len(class_performance.T)): ax01.plot(epsilons,class_performance[:,i],'-o',color=pallette2[i])
ax01.set_xscale('log')
ax01.grid('on')
ax01.set_ylim(0,1)

ax02 = fig.add_subplot(gs[:2,4:])
ax02.plot(epsilons,tree_distance,'-o',color=pallette2[0])
ax02.set_xscale('log')
ax02.grid('on')

axs_imgs = []
for i in range(3):
    attack_id = i+5
    axs_imgs.append(fig.add_subplot(gs[2+2*i:2+2*(1+i),:2]))
    axs_imgs[-1].imshow(orig_images1[attack_id],cust_cmap)
    axs_imgs[-1].axis('off')
    plt.title('$\epsilon$ = '+str(round(epsilons[attack_id],2))+'; label = '+str(orig_labels1[attack_id].item()))
    
    
    axs_imgs.append(fig.add_subplot(gs[2+2*i:2+2*(1+i),2:4]))
    axs_imgs[-1].imshow(pert_images1[attack_id],cust_cmap)
    axs_imgs[-1].axis('off')
    plt.title('predicted label = '+str(pert_labels1[attack_id].item()))

    axs_imgs.append(fig.add_subplot(gs[2+2*i:2+2*(1+i),4:6]))
    axs_imgs[-1].imshow(orig_images2[attack_id],cust_cmap)
    axs_imgs[-1].axis('off')
    plt.title('$\epsilon$ = '+str(round(epsilons[attack_id],2))+'; label = '+str(orig_labels2[attack_id].item()))

    axs_imgs.append(fig.add_subplot(gs[2+2*i:2+2*(1+i),6:]))
    axs_imgs[-1].imshow(pert_images2[attack_id],cust_cmap)
    axs_imgs[-1].axis('off')
    plt.title('predicted label = '+str(pert_labels2[attack_id].item()))


plt.tight_layout()

# plt.savefig('../Figures/MNIST_Iso'+attack_type+str(Iso_coef)+'.png',dpi=1000)




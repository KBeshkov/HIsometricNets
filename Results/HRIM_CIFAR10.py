#%% CIFAR10 run
import sys,os
sys.path.append(os.getcwd())
sys.path.append(os.getcwd()[:-7]+'Analysis')
sys.path.append('/scratch/users/constb/PyTorch_CIFAR10/') #load pretrained models
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
import seaborn as sns
from torchvision import datasets, transforms
plt.style.use('default')
plt.rcParams['font.family'] = 'Arial'
pallette2 = sns.color_palette('colorblind')
cust_cmap =sns.color_palette("rocket", as_cmap=True)
sys.path.append(os.getcwd())
sys.path.append(os.getcwd()[:-7]+'Analysis')
from Algorithms import *
from metrics import *
from ML_models import *
from torchviz import make_dot

#%%
sys.path.append("/Users/kosio/Repos/PyTorch_CIFAR10/")
from cifar10_models.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn

# Pretrained model
# feature_model = vgg13_bn(pretrained=True)
# feature_model.eval() # for evaluation

feature_model = CNN_Features(first_layer_output=128).double()
#%%
Iso_coef = 0.01

transform=transforms.Compose([transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))])

data_train = datasets.CIFAR10('/Users/kosio/Repos/HIsometricNets/Data/', train=True, transform=transform, download=True)
data_test = datasets.CIFAR10('/Users/kosio/Repos/HIsometricNets/Data/', train=False, transform=transform, download=True)


data_train_hierarchy = HierarchicalDataset(data_train,conv=True)
data_test_hierarchy = HierarchicalDataset(data_test,conv=True)

partition = [[[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]]]

data_part_train = data_train_hierarchy.__hierarchy__(partition)
data_part_test = data_test_hierarchy.__hierarchy__(partition)

hierarchical_trainloader = torch.utils.data.DataLoader(data_train_hierarchy, batch_size=100, shuffle=True)
hierarchical_testloader = torch.utils.data.DataLoader(data_test_hierarchy, batch_size=100, shuffle=True)
  
precomputed_dmat = 'standard'
if precomputed_dmat == True:
    CIFAR_dmat = torch.load('E:\Data\HIsom_Data\CIFAR10_dmat_features.pt').double() #Precomputed distance matrix between MNIST images
elif precomputed_dmat=='standard':
    CIFAR_dmat='standard'
elif precomputed_dmat=='csv':
    CIFAR_dmat = 'csv:E:/Data/HIsom_Data/CIFAR10_dmat.csv'
else:
    CIFAR_dmat='iterate'
res = int(2*(data_train[0][0].size()[-1]/2)**2)*32

#%% Train Model
nlayers = 3
nclass = len(partition)
n_nrns = 1000
network_params = {'n_neurons': n_nrns, 'n_inputs': res,'n_classes': 10,
                  'projection': [],'weights': [],'n_layers': nclass*nlayers,'feature_model':feature_model}
HRIM = HierarchicalRepresentationNetwork(NN_class(**network_params).double().to('cpu'),savepoints=25,device='cpu')

#%%
K = 1
costs = [[nn.CrossEntropyLoss(),MetricPreservationLoss(Iso_coef, K).Loss]]

HRIM_data = HRIM.train(hierarchical_trainloader, costs, dmat=CIFAR_dmat,epochs=10)

#%% Test model
HRIM_test = HRIM.test(hierarchical_testloader)
print(HRIM_test)

#%% Testing adverserial attacks

#define epsilon ranges
epsilons = np.logspace(-2,0,10)


costs = [[nn.CrossEntropyLoss(),MetricPreservationLoss(Iso_coef, 1).Loss]]

attack_type = 'fgsm'
adv_attacks = []
for e in epsilons:
    adv_attacks.append(HRIM.hierarchical_test_attack(hierarchical_testloader,costs,
                                                     epsilon=e, attack_method=attack_type,featurize=True))

class_performance = np.hstack([adv_attacks[i][0] for i in range(len(epsilons))]).T
orig_images1 = [adv_attacks[i][1][-1][2][5].T for i in range(len(epsilons))]
orig_labels1 = [adv_attacks[i][1][-1][0][5] for i in range(len(epsilons))]
orig_images2 = [adv_attacks[i][1][-1][2][3].T for i in range(len(epsilons))]
orig_labels2 = [adv_attacks[i][1][-1][0][3] for i in range(len(epsilons))]
pert_images1 = [adv_attacks[i][1][-1][3][5].T for i in range(len(epsilons))]
pert_labels1 = [adv_attacks[i][1][-1][1][5] for i in range(len(epsilons))]
pert_images2 = [adv_attacks[i][1][-1][3][3].T for i in range(len(epsilons))]
pert_labels2 = [adv_attacks[i][1][-1][1][3] for i in range(len(epsilons))]
saliency_maps1 = [saliency_map(torch.Tensor(orig_images1[i]).T[None], HRIM.model, orig_labels1[i],threshold=0.3) for i in range(len(epsilons))]
saliency_maps2 = [saliency_map(torch.Tensor(orig_images2[i]).T[None], HRIM.model, orig_labels2[i],threshold=0.3) for i in range(len(epsilons))]

#%%
fig, ax = plt.subplots(6,len(epsilons))


for j in range(len(epsilons)):
    ax[0,j].imshow(orig_images1[j])
    ax[1,j].imshow(orig_images1[j])
    ax[1,j].pcolormesh(saliency_maps1[j][0].T,alpha=1,cmap='coolwarm',shading='auto')
    ax[2,j].imshow(pert_images1[j])
    ax[3,j].imshow(orig_images2[j])
    ax[4,j].imshow(orig_images2[j])
    ax[4,j].pcolormesh(saliency_maps2[j][0].T,alpha=1,cmap='coolwarm',shading='auto')
    ax[5,j].imshow(pert_images2[j])
for cur_ax in ax:
    for cur_ax in cur_ax:
        cur_ax.axis('off')

fig.tight_layout()

fig.savefig('../Figures/images_'+str(Iso_coef)+'_CIFAR.png',dpi=500)
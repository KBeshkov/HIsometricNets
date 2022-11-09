#%% CIFAR10 run
import sys,os
sys.path.append(os.getcwd())
sys.path.append(os.getcwd()[:-7]+'Analysis')
sys.path.append('/Users/constb/Repos/PyTorch_CIFAR10/') #load pretrained models
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


#%%
from cifar10_models.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn

# Pretrained model
feature_model = vgg13_bn(pretrained=True)
feature_model.eval() # for evaluation
#%%
Iso_coef = 1

transform=transforms.Compose([transforms.ToTensor()])

data_train = datasets.CIFAR10('/Users/constb/Data/MNIST/', train=True, transform=transform)#, download=True)
data_test = datasets.CIFAR10('/Users/constb/Data/MNIST/', train=False, transform=transform)#, download=True)


data_train_hierarchy = HierarchicalDataset(data_train,conv=True)
data_test_hierarchy = HierarchicalDataset(data_test,conv=True)

partition = [[[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]]]

data_part_train = data_train_hierarchy.__hierarchy__(partition)
data_part_test = data_test_hierarchy.__hierarchy__(partition)

hierarchical_trainloader = torch.utils.data.DataLoader(data_train_hierarchy, batch_size=200, shuffle=True)
hierarchical_testloader = torch.utils.data.DataLoader(data_test_hierarchy, batch_size=200, shuffle=True)
  
precomputed_dmat = True
if precomputed_dmat == True:
    CIFAR_dmat = torch.load(os.getcwd()[:-8]+'/Data/CIFAR10_dmat_features.pt').double() #Precomputed distance matrix between MNIST images
elif precomputed_dmat=='standard':
    CIFAR_dmat='standard'
else:
    CIFAR_dmat='iterate'
res = int(2*(data_train[0][0].size()[-1]/2)**2)

#%% Train Model
nlayers = 5
nclass = len(partition)
n_nrns = 100
network_params = {'n_neurons': n_nrns, 'n_inputs': res,'n_classes': 10,
                  'projection': [],'weights': [],'n_layers': nclass*nlayers,'feature_model':feature_model.features}
HRIM = HierarchicalRepresentationNetwork(NN_class(**network_params).double().to('cpu'),savepoints=25)

#%%
HRIM_test = HRIM.test(hierarchical_testloader)
print('Before training:' +str(HRIM_test))
K = 10
costs = [[nn.CrossEntropyLoss(),MetricPreservationLoss(Iso_coef, K).Loss]]

HRIM_data = HRIM.train(hierarchical_trainloader, costs, dmat=CIFAR_dmat,epochs=1)

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
                                                     epsilon=e, attack_method=attack_type))

class_performance = np.hstack([adv_attacks[i][0] for i in range(len(epsilons))]).T
orig_images1 = [np.reshape(adv_attacks[i][1][-1][2][5],[28,28]) for i in range(len(epsilons))]
orig_labels1 = [adv_attacks[i][1][-1][0][5] for i in range(len(epsilons))]
orig_images2 = [np.reshape(adv_attacks[i][1][-1][2][3],[28,28]) for i in range(len(epsilons))]
orig_labels2 = [adv_attacks[i][1][-1][0][3] for i in range(len(epsilons))]
pert_images1 = [np.reshape(adv_attacks[i][1][-1][3][5],[28,28]) for i in range(len(epsilons))]
pert_labels1 = [adv_attacks[i][1][-1][1][5] for i in range(len(epsilons))]
pert_images2 = [np.reshape(adv_attacks[i][1][-1][3][3],[28,28]) for i in range(len(epsilons))]
pert_labels2 = [adv_attacks[i][1][-1][1][3] for i in range(len(epsilons))]

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
plt.style.use('default')
plt.rcParams['font.family'] = 'Arial'
pallette2 = sns.color_palette('colorblind')
cust_cmap =sns.color_palette("rocket", as_cmap=True)
sys.path.append(os.getcwd())
sys.path.append(os.getcwd()[:-7]+'Analysis')
from Algorithms import *
from metrics import *
from ML_models import *

Iso_coef = 0.1

transform=transforms.Compose([transforms.ToTensor()])

data_train = datasets.MNIST('/Users/constb/Data/MNIST/', train=True, transform=transform)#, download=True)
data_test = datasets.MNIST('/Users/constb/Data/MNIST/', train=False, transform=transform)#,download=True)

data_train_hierarchy = HierarchicalDataset(data_train)
data_test_hierarchy = HierarchicalDataset(data_test)

partition = [[[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]]]

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

K = 10
costs = [[nn.CrossEntropyLoss(),MetricPreservationLoss(Iso_coef, K).Loss]]

HRIM_data = HRIM.train(hierarchical_trainloader, costs, dmat=MNIST_dmat,epochs=5)

#%% Test model
HRIM_test = HRIM.test(hierarchical_testloader)
print(HRIM_test)

#%% Testing adverserial attacks

#define epsilon ranges
epsilons = np.logspace(-2,0,10)


costs = [[nn.CrossEntropyLoss(),MetricPreservationLoss(Iso_coef, K).Loss]]

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

#%% Degree of error measured in terms of the distance between the MNIST predicted labels
orig_labels = [adv_attacks[i][1][-1][0] for i in range(len(epsilons))]
pert_labels = [adv_attacks[i][1][-1][1] for i in range(len(epsilons))]
distances = [torch.sum(torch.abs(orig_labels[i]-pert_labels[i])).detach().numpy() for i in range(len(epsilons))]

np.save('../Data/fgsm_temp/class_curves_'+attack_type+str(Iso_coef)+'.npy',class_performance[:,-1])
# np.save('../Data/FGSM_tree_nomax/distances_'+attack_type+str(Iso_coef)+'.npy',np.array(tree_distance))


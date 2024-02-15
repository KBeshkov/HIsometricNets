#%% Ican2021 run
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
from Ican_2021_Dataset import *
from torchviz import make_dot

#%%
sys.path.append("/Users/kosio/Repos/PyTorch_CIFAR10/")
from cifar10_models.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn

# Pretrained model
# feature_model = vgg13_bn(pretrained=True)
# feature_model.eval() # for evaluation

feature_model = CNN_Features(first_layer_output=128).double()
#%%
Iso_coef = 0.0001

transform=transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

data_train = InatCustom('/Users/kosio/Repos/HIsometricNets/Data/inat-2021/semi-inat-2021/train/train/', transform=transform)
data_test = InatCustom('/Users/kosio/Repos/HIsometricNets/Data/inat-2021/semi-inat-2021/test/test/', transform=transform)


data_train_hierarchy = HierarchicalDataset(data_train,conv=True,scale_img=False)
data_test_hierarchy = HierarchicalDataset(data_test,conv=True,scale_img=False)

partition = [[list(set(data_train.kingdom_labels[key])) for key in data_train.kingdom_labels],
             [list(set(data_train.phylum_labels[key])) for key in data_train.phylum_labels],
             [list(set(data_train.class_labels[key])) for key in data_train.class_labels],
             [list(set(data_train.order_labels[key])) for key in data_train.order_labels],
             [list(set(data_train.family_labels[key])) for key in data_train.family_labels],
             [list(set(data_train.genus_labels[key])) for key in data_train.genus_labels],
             [[i] for i in set(data_train.targets)]]

data_part_train = data_train_hierarchy.__hierarchy__(partition)
data_part_test = data_test_hierarchy.__hierarchy__(partition)

hierarchical_trainloader = torch.utils.data.DataLoader(data_train_hierarchy, batch_size=100, shuffle=True)
hierarchical_testloader = torch.utils.data.DataLoader(data_test_hierarchy, batch_size=100, shuffle=False)
  
precomputed_dmat = 'standard'
if precomputed_dmat=='standard':
    Ican_dmat='standard'
else:
    Ican_dmat='iterate'
res = int(2*(data_train[0][0].size()[-1]/2)**2)*2048*8

#%%
nlayers = 3
nclass = len(partition)
n_nrns = 200
network_params = {'n_neurons': n_nrns, 'n_inputs': res,'n_classes': len(set(data_train.targets)),
                  'projection': [],'weights': [],'n_layers': nclass*nlayers,'feature_model':feature_model}
HRIM = HierarchicalRepresentationNetwork(NN_class(**network_params).double().to('cpu'),savepoints=25,device='cpu')

#%%Train or load pretrained model
K = 1
costs = [[nn.CrossEntropyLoss(),MetricPreservationLoss(Iso_coef, K).Loss] for i in range(len(partition))]

# HRIM.model.load_state_dict(torch.load("/Users/kosio/Data/HRIM_ICAN.pt",map_location = torch.device('cpu')))
HRIM_data = HRIM.train(hierarchical_trainloader, costs, dmat=Ican_dmat,epochs=20)

#%% Test model
HRIM_test = HRIM.test(hierarchical_testloader)
print(HRIM_test)

#%% Testing adverserial attacks
first_n = 100
orig_images = hierarchical_testloader.dataset.x[:first_n]
labels_images = [hierarchical_testloader.dataset.labels[i][:first_n] for i in range(len(hierarchical_testloader.dataset.labels))]
saliency_maps = HRIM.saliency_map(orig_images, labels_images, threshold=0.01)


#%%
n_images = 3
rand_images = np.random.choice(np.arange(0,len(orig_images)),replace=False,size=n_images)
hierarchies = ["kingdom","phylum","class","order","family","genus","species"]
fig, ax = plt.subplots(n_images,len(partition))

for i,img_id in enumerate(rand_images):
    for j in range(len(partition)):
        # ax[i,j].imshow(orig_images[img_id].detach().numpy().T)
        ax[i,j].pcolormesh(saliency_maps[j][img_id].T,alpha=0.4,cmap='coolwarm',shading='auto')
        if i==0:
            ax[i,j].set_title(hierarchies[j])
    for cur_ax in ax:
        for cur_ax in cur_ax:
            cur_ax.axis('off')

#%%Dataloader class
import sys,os
sys.path.append(os.getcwd())
sys.path.append(os.getcwd()[:-7]+'Analysis')
from Algorithms import *
from metrics import *
from ML_models import *
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
            
res = 100
cdd = CatDogDataset('/Users/constb/Data/ImageNet',torch.nn.Sequential(Grayscale(),Resize([res,res])))    
set_hier = cdd.__hierarchy__()         
        

#%%
# from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt

dataloader = DataLoader(cdd, batch_size=1,
                        shuffle=True)#, num_workers=0)

# for c,i in enumerate(dataloader):
#     if c%50==0:
#         print(c)
#         plt.figure()
#         plt.imshow(i[0][0],cmap='gray')
        

#%%
full_data = cdd.get_all()
train_size = int(0.01 * len(full_data[0]))
test_size = len(full_data[0]) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_data[0], [train_size, test_size],generator=torch.Generator().manual_seed(12))
train_labels1, test_labels1 = torch.utils.data.random_split(full_data[1][0], [train_size, test_size],generator=torch.Generator().manual_seed(12))
train_labels2, test_labels2 = torch.utils.data.random_split(full_data[1][1], [train_size, test_size],generator=torch.Generator().manual_seed(12))

train_dataset = torch.tensor(torch.reshape(torch.tensor(np.stack(train_dataset)),[len(train_dataset),res**2])).double()
labels = [np.array(train_labels1,dtype=np.float64),np.array(train_labels2,np.float64)]
#%%
nlayers = 5
nclass = len(labels)
n_nrns = 1000
network_params = {'n_neurons': n_nrns, 'n_inputs': res**2,'n_classes': 72,
                  'projection': [],'weights': [],'n_layers': nclass*nlayers}
HRIM = HierarchicalRepresentationNetwork(NN_class(**network_params).double().to('cpu'),savepoints=25)

costs = [[nn.CrossEntropyLoss(),MetricPreservationLoss(1e5, 0).Loss] for i in range(nlayers)]

HRIM_data = HRIM.train(train_dataset, labels, costs, epochs=2000)





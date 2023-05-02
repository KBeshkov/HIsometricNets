# Hierarchical Representations through locally isomorphic embeddings
import sys, os

sys.path.append(os.getcwd())
sys.path.append(os.getcwd()[:-7] + "Analysis")
from Algorithms import *
from metrics import *
from ML_models import *
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

plt.rcParams["font.family"] = "Arial"
pallette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
pallette2 = sns.color_palette("colorblind")
plt.style.use("default")

#%%
nlayers = 4
n_nrns = 20

N = 40
partition = [
    [[0, 1, 2, 3], [4, 5, 6, 7]],
    [[0, 1], [2, 3], [4, 5], [6, 7]],
    [[0], [1], [2], [3], [4], [5], [6], [7]],
]
labels = binary_hierarchical_labeling([2, 4, 8], N**2)

noise = 1e-3
R, r = 0.67, 0.33
hdim = 3
Mfld = torch.tensor(Manifold_Generator().T2(N, R, r)).T[:,:,None]#@torch.randn(3,hdim).double())[:, :, None]
Mfld = torch.utils.data.TensorDataset(
    Mfld + noise * torch.randn(Mfld.size()), torch.tensor(labels[-1])
)
Mfld = MfldDataset(Mfld)


nclass = len(labels)
data_hierarchy = HierarchicalDataset(Mfld, scale_img=False)


data_part = data_hierarchy.__hierarchy__(partition)

hierarchical_loader = torch.utils.data.DataLoader(
    data_hierarchy, batch_size=N**2, shuffle=False
)


network_params = {
    "n_neurons": n_nrns,
    "n_inputs": hdim,
    "n_classes": 8,
    "projection": [],
    "weights": [],
    "n_layers": nlayers * nclass,
}

HRIM = HierarchicalRepresentationNetwork(
    NN_class(**network_params).double(), savepoints=25
)

costs = [
    [nn.CrossEntropyLoss(), MetricPreservationLoss(10, 1).Loss],
    [nn.CrossEntropyLoss(), MetricPreservationLoss(10, 1).Loss],
    [nn.CrossEntropyLoss(), MetricPreservationLoss(10, 1).Loss],
]

HRIM_data = HRIM.train(hierarchical_loader, costs, epochs=5000, plot=True)
#%%
HRIM_test = HRIM.test(hierarchical_loader)
performance = HRIM_test[-1]
print(performance)
#%%
dmats = []
reducer = umap.UMAP(n_neighbors=40, min_dist=0.25, n_components=3)
fig = plt.figure(dpi=300)
for i in range(nclass):
    dmats.append(pairwise_distances(HRIM_data[0][0][i][0].detach().numpy()))
    mfld_pred = reducer.fit_transform(HRIM_data[0][0][i][0].detach().numpy()).T
    ax1 = fig.add_subplot(1, nclass, i + 1, projection="3d")
    for j in np.unique(labels[i]).astype(int):
        ax1.scatter(
            mfld_pred[0, labels[i] == j],
            mfld_pred[1, labels[i] == j],
            mfld_pred[2, labels[i] == j],
            s=0.5,
            alpha=0.7,
            color=pallette2[j],
        )
        ax1.axis("off")
        ax1.grid("off")
plt.tight_layout()
# plt.savefig('../Figures/class_dec_Metric_layer_'+str(i+1)+'.png',transparent=True,dpi=1000)

plt.figure()
for i in range(len(labels)):
    plt.subplot(1,len(labels),i+1)
    plt.imshow(dmats[i])

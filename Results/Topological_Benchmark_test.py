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
pallette2 = sns.color_palette('dark')
plt.style.use('default')#../misc/report.mplstyle')

#%%
network_params = {'n_neurons': 100, 'n_inputs': 3,'n_classes': 5,
                  'projection': [],'weights': [],'n_layers': 2}
settings = {'nperms': 1, 'nsamples': 15, 'npoints': 16,
            'dim': 1, 'mfld': Manifold_Generator().T2, 'mfld_noise':0,
            'phom_params': [pairwise_distances, True, 1, 50],
            'model': NN_class(**network_params).double(), 'nclasses': 5,
            'criterion': MetricPreservationLoss(lambda1=10,lambda2=0).Loss,
            'model_params': [500, 1], 'mfld_args': [1, 0.66],'local': True}
TopBenchmark = TopologicalBenchmark(**settings)

#%%
# test = TopBenchmark.full_test()
# print('Pvalue = '+str(test[0][3][0]))

mfld = Manifold_Generator().T2(25,1,0.66)
model_mfld = TopBenchmark.model(torch.tensor(mfld))[0].detach().numpy().T

Phom = Persistent_Homology()

pdiags1 = Phom(mfld.T,pairwise_distances, True, 1, 50)[1]
pdiags2 = Phom(model_mfld.T,pairwise_distances, True, 1, 50)[1]
plands1 = persim.landscapes.PersLandscapeApprox(dgms=pdiags1,hom_deg=1)
plands2 = persim.landscapes.PersLandscapeApprox(dgms=pdiags2,hom_deg=1)
[avg_pland1_snapped, avg_pland2_snapped] = persim.landscapes.snap_pl([plands1, plands2])
diff_pl = avg_pland1_snapped - avg_pland2_snapped

#%%
ax_ = plt.subplot(1,1,1,projection='3d')
ax_.scatter(mfld[0],mfld[1],mfld[2],s=10,alpha=0.4,c='k')
ax_.axis('off') 
ax_.grid('off')
# plt.savefig('../Figures/mfld_orig.png',bbox_inches="tight",dpi=1000,transparent=True)

plt.figure()
reducer = umap.UMAP(n_neighbors=45,min_dist=0.75,n_components=3)
red_mfld = reducer.fit_transform(model_mfld.T).T
ax_ = plt.subplot(1,1,1,projection='3d')
ax_.scatter(red_mfld[0],red_mfld[1],red_mfld[2],s=10,alpha=0.4,c='k')
ax_.axis('off') 
ax_.grid('off')
# plt.savefig('../Figures/mfld_model.png',bbox_inches="tight",dpi=1000,transparent=True)


persim.landscapes.plot_landscape(plands1,500)
# plt.savefig('../Figures/mfld_plnd.png',bbox_inches="tight",dpi=1000,transparent=True)
persim.landscapes.plot_landscape(plands2,500)
# plt.savefig('../Figures/model_plnd.png',bbox_inches="tight",dpi=1000,transparent=True)
persim.landscapes.plot_landscape(diff_pl,500)
# plt.savefig('../Figures/diff_plnd.png',bbox_inches="tight",dpi=1000,transparent=True)

#%%Plot baseline vs regularizers
CSE_vals = np.logspace(-4,2,5)
Metric_vals = np.logspace(-2,4,5)
X,Y = np.meshgrid(CSE_vals,Metric_vals)
top_heatmap = np.zeros([len(CSE_vals),len(Metric_vals)])
top_heatmaps = [np.zeros([len(CSE_vals),len(Metric_vals)]) for i in range(4)]
top_pvals = [np.zeros([len(CSE_vals),len(Metric_vals)]) for i in range(4)]
perform = [np.zeros([len(CSE_vals),len(Metric_vals)]) for i in range(4)]
# settings['local'] = False

for count1,l1 in enumerate(CSE_vals):
    for count2,l2 in enumerate(Metric_vals):
        settings['model_params'][1] = l1
        settings['criterion'] = MetricPreservationLoss(lambda1=l2,lambda2=0).Loss
        TopBenchmark = TopologicalBenchmark(**settings)
        try:
            top_heatmap = TopBenchmark.full_test()
        except:
            top_heatmap = TopBenchmark.full_test()
        for i in range(len(top_heatmap[0])):
            top_heatmaps[i][count1,count2] = top_heatmap[0][i][1]
            top_pvals[i][count1,count2] = top_heatmap[0][i][0]
            perform[i][count1,count2] = np.mean(top_heatmap[1])
        print((count1,count2))
        # plt.figure()
        # plt.hist(histogram,bins=60)

top_heatmaps_norm = [((hmap)/np.max(hmap))for hmap in top_heatmaps]
performance_norm = [((perf)/np.max(perf)) for  perf in perform]
top_perf_diff_norm = [performance_norm[i]-top_heatmaps_norm[i] for i in range(len(perform))]
#%%
import matplotlib as mpl
def create_classes_plot(heatmap,X,Y,xclass,yclass,vmin,vmax,xlabel,ylabel,title,savename=''):
    fig = plt.figure()   
    axs = fig.subplots(xclass,yclass)
    count = 0
    for i in range(2):
        for j in range(2):
            im = axs[i,j].contourf(X,Y,heatmap[count].T,10,cmap="RdBu_r",alpha=1)#,vmin=vmin,vmax=vmax)
            # axs[i,j].imshow(top_heatmaps[count])
            axs[i,j].set_yscale('log')
            axs[i,j].set_xscale('log')
            if j==0 and i==0:
                axs[i,j].set_ylabel(ylabel,font='Arial') 
            if i==1 and j==0:
                axs[i,j].set_xlabel(xlabel,font='Arial')
            axs[i,j].text(10**1,10**2,'Label '+str(count+1),fontsize=6,font='Arial')
            count += 1
    # plt.suptitle(title)
    plt.tight_layout()
    fig.colorbar(im, ax=axs.ravel().tolist(),location='bottom',aspect=50)
    # if len(savename)>0:
        # plt.savefig('../Figures/' + savename + '.png',bbox_inches="tight",dpi=1000,transparent=True)

create_classes_plot(top_heatmaps_norm,X,Y,2,2,0,1,'CSE Loss','Metric Loss','Topological Destruction','top_heatmap')
create_classes_plot(performance_norm,X,Y,2,2,0,1,'CSE Loss','Metric Loss','Performance','perform_heatmap')
create_classes_plot(top_perf_diff_norm,X,Y,2,2,-1,1,'CSE Loss','Metric Loss','Difference','diff_heatmap')


#ML experiments
import sys,os
sys.path.append(os.getcwd())
sys.path.append(os.getcwd()[:-7]+'Analysis')
from Algorithms import *
from metrics import *
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import pickle

plt.rcParams['font.family'] = 'Arial'
plt.rcParams.update({'font.size': 14})
pallette = plt.rcParams['axes.prop_cycle'].by_key()['color']
pallette2 = sns.color_palette('colorblind')
plt.style.use('default')#../misc/report.mplstyle')
#%% generate manifolds
N = 40
mfld_gen = Manifold_Generator()

#%%ML
from ML_models import *

N_net = 50
n_layers = 4
test_add = 20 #determines how many extra samples to add
f_M = lambda x: x
data = f_M(mfld_gen.T2(N,1,0.66))

D = len(data)
noise_std = 0.01
dat_test = f_M(mfld_gen.T2(N+test_add,1,0.66))+noise_std*np.random.randn(len(data),(N+test_add)**2)

#define the classes on the manifold
nclass = 9
classes = np.linspace(0,N**2,nclass).astype(int)
test_classes = np.linspace(0,(test_add+N)**2,nclass).astype(int)
labels = 0.1*np.ones(N**2)
labels_test = 0.1*np.ones((N+test_add)**2)
count=0
perturb_weight = 1e-3 #choose whether to perturb the points slightly to avoid highly overlapping datapoints 
for i in range(nclass-1):
    labels[classes[i]:classes[i+1]] = count
    train_perturb = perturb_weight*np.matlib.repmat(np.mean(data[:,labels==count],1),sum(labels==count),1).T
    data[:,labels==count] = data[:,labels==count] + train_perturb
    labels_test[test_classes[i]:test_classes[i+1]] = count
    test_perturb = perturb_weight*np.matlib.repmat(np.mean(dat_test[:,labels_test==count],1),sum(labels_test==count),1).T
    dat_test[:,labels_test==count] = dat_test[:,labels_test==count]+test_perturb
    count += 1
train_data = [torch.tensor(data),torch.tensor(labels)]
test_data = [torch.tensor(dat_test),torch.tensor(labels_test)]

#%%
model = NN_class(N_net,D,len(np.unique(labels)),[],[],n_layers).double()
dmat = torch.cdist(train_data[0].T,train_data[0].T)
init_run = torch.tensor(predict(model,train_data[0].T)[1])
optimizer = optim.Adam(model.parameters())
epoch_n = 10000

train_type = 'Metric'
if train_type=='Metric':
    loss_weights = [1,1]
elif train_type =='CSE':
    loss_weights = [1,0]
    
train_params = {'net':model,'train_dat':train_data,'optimizer':optimizer,'criterion_weights':loss_weights,
                'epochs':epoch_n,'batch_sz':len(train_data[0].T),'dmat':dmat}

loss_curve = train(**train_params)


plt.figure()
for count,i in enumerate(loss_curve[:-1]): plt.plot(i,linewidth=2,color=pallette2[count])
plt.legend(['Total','Cross-Entropy','Metric'])
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.grid('on')
plt.xscale('log')
plt.yscale('log')
# plt.savefig('../Figures/Loss'+train_type+'.png', transparent=True,dpi=1000)
#%%
predictions = predict(model,test_data[0].T)
predicted_labels = np.argmax(predictions[1],1)
performance = np.sum(labels_test==predicted_labels)/len(labels_test)
print(performance)

#%%
fig = plt.figure(dpi=300,figsize=(2,2))
        
ax1 = fig.add_subplot(1,1,1,projection='3d')               
for i in range(nclass-1):
    ax1.scatter(dat_test[0,labels_test==i],dat_test[1,labels_test==i],dat_test[2,labels_test==i],
                s=.5,alpha=0.7,color=pallette2[i])
ax1.set_axis_off()
# plt.savefig('../Figures/T2_class.png',transparent=True,dpi=1000)

pca_pred = umap.UMAP(n_neighbors=40,min_dist=1,n_components=3).fit_transform(predictions[0]).T

fig=plt.figure(dpi=300,figsize=(2,2))
ax2 = fig.add_subplot(1,1,1,projection='3d')
for i in range(nclass-1):
    ax2.scatter(pca_pred[0,labels_test==i],pca_pred[1,labels_test==i],pca_pred[2,labels_test==i],
                s=.5,alpha=0.7,color=pallette2[i])
ax2.set_axis_off()
plt.tight_layout()
# plt.savefig('../Figures/T2_class_dec_'+train_type+'.png',transparent=True,dpi=1000)


#%%Compare the metric and topological distortion of different regularizers
network_params = {'n_neurons': N_net, 'n_inputs': len(data),'n_classes': nclass,
                  'projection': [],'weights': [],'n_layers': n_layers}
settings = {'nperms': 1, 'nsamples': 10, 'npoints': 16,
            'dim': 1, 'mfld': Manifold_Generator().T2, 'mfld_noise':0,
            'phom_params': [pairwise_distances, True, 1, 50],
            'model': NN_class(**network_params).double(), 'nclasses': nclass,
            'criterion': MetricPreservationLoss(lambda1=10,lambda2=0).Loss,
            'model_params': [100, 1,500,10**2], 'mfld_args': [2, 1],'local': True}
TopBenchmark = TopologicalBenchmark(**settings)

CSE_vals = np.logspace(-4,2,6)
Metric_vals = np.logspace(-2,4,6)
X,Y = np.meshgrid(CSE_vals,Metric_vals)
top_heatmap = np.zeros([len(CSE_vals),len(Metric_vals)])
top_heatmaps = [np.zeros([len(CSE_vals),len(Metric_vals)]) for i in range(nclass-1)]
top_pvals = [np.zeros([len(CSE_vals),len(Metric_vals)]) for i in range(nclass-1)]
perform = [np.zeros([len(CSE_vals),len(Metric_vals)]) for i in range(nclass-1)]

for count1,l1 in enumerate(CSE_vals):
    for count2,l2 in enumerate(Metric_vals):
        settings['model_params'][0] = l1
        settings['model_params'][1] = l2
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
            im = axs[i,j].contourf(X,Y,heatmap[count].T,6,cmap=sns.color_palette("vlag", as_cmap=True),alpha=1)#,vmin=vmin,vmax=vmax)
            # im=axs[i,j].imshow(top_heatmaps[count])
            axs[i,j].set_yscale('log')
            axs[i,j].set_xscale('log')
            if j==0 and i==0:
                axs[i,j].set_ylabel(ylabel,font='Arial') 
            if i==1 and j==0:
                axs[i,j].set_xlabel(xlabel,font='Arial')
            axs[i,j].text(10**1,10**2,str(count+1),fontsize=6,font='Arial')
            count += 1
    plt.suptitle(title)
    plt.tight_layout()
    fig.colorbar(im, ax=axs.ravel().tolist(),location='bottom',aspect=50)
    if len(savename)>0:
        plt.savefig('../Figures/' + savename + '.png',bbox_inches="tight",dpi=1000,transparent=True)

create_classes_plot(top_heatmaps_norm,X,Y,2,2,0,1,'CSE Loss','Metric Loss','Topological Destruction','top_heatmap')
create_classes_plot(performance_norm,X,Y,2,2,0,1,'CSE Loss','Metric Loss','Performance','perform_heatmap')
create_classes_plot(top_perf_diff_norm,X,Y,2,2,-1,1,'CSE Loss','Metric Loss','Performance-Topological Destruction','diff_heatmap')

#%% Manifolds through training
reducer =  umap.UMAP(n_neighbors=50,min_dist=0.75,n_components=3)
anim_data = [reducer.fit_transform(img.detach().numpy()) for img in loss_curve[-1]]

#%%Animations
def animate(i,embedding,labels):
    ax.cla()
    ax.scatter(embedding[i][:,0],embedding[i][:,1],embedding[i][:,2],s=10,c=labels,cmap=sns.color_palette("vlag", as_cmap=True))

fig = plt.figure(figsize=(6,6),dpi=200,constrained_layout=True)
ax = plt.subplot(111,projection='3d')
ax.axis('off')
anim = animation.FuncAnimation(fig, animate, blit=False, frames = 60,fargs=(anim_data,labels),interval=2000)
anim.save('../Figures/class_Metric.gif', writer='imagemagick', fps=60)
    
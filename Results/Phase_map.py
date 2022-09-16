#phase diagram of 
import sys,os
sys.path.append(os.getcwd())
sys.path.append(os.getcwd()[:-7]+'Analysis')
from Algorithms import *
from metrics import *
import matplotlib.pyplot as plt
# plt.rcParams['font.family'] = 'Arial'

#%% generate manifolds
N = 20
mfld_gen = Manifold_Generator()

S1 = mfld_gen.S1(N**2, 1)
S2 = np.random.randn(6,N**2)#mfld_gen.S2(N,1)#
T2 = mfld_gen.T2(N,1,0.66)


#%% Generate weight matrices with a given spectrum
S2_H2_net_ = []
for k in range(1):
    N_net = 81#len(S2.T)#
    D = len(S2)
    n_mats = 40
    n_proj = 1
    
    Wmats = []
    decay_consts = np.linspace(0,1,n_mats)#np.logspace(-2,1,n_mats)#np.logspace(1.5, 3.5, n_mats)#
    coord_num = []
    g = 3
    lower_coord = 0#.05
    proj_operators = [0]*n_mats
    orient_const = np.linspace(0,1,n_proj)
    x = np.arange(1,N_net+1)#np.linspace(0,lower_coord,N_net)#
    alt_eig = np.ones(N_net)
    # alt_eig[np.arange(1,N_net,2)]=-alt_eig[np.arange(1,N_net,2)]
    eiglist = []
    
    plt.figure(dpi=300,figsize=(6,10))
    count = 0
    for i in decay_consts:
        eigvals = g*x**-i#-(1/lower_coord)*(g-lower_coord)*(x-lower_coord)*(np.exp(-i*x))+lower_coord#
        # Wmats.append(gen_weight_mat(N_net,N_net,g=0,svd='qr_decomp',eigenvals=eigvals,zero=0)[0])#g*ortho_group.rvs(N_net))#g*np.exp(-i*pairwise_distances(S2.T)**2))#
        Wmats.append(g*((1-i)*ortho_group.rvs(N_net)+i*2*np.random.randn(N_net,N_net)))
        eigval, eigvect = np.linalg.eig(Wmats[-1])
        sing_vals = np.linalg.svd(Wmats[-1])[1]
        eiglist.append((np.real(eigval),np.imag(eigval)))
        # plt.plot(np.real(eigval),np.imag(eigval),'.')#sing_vals)
        # plt.plot(x,np.abs(eigval),'brown',alpha=1/n_mats+count*(1/n_mats),label=str(round(i,1)))
        plt.plot(x,sing_vals,'brown',alpha=1/n_mats+count*(1/n_mats),label=str(round(i,1)))
        coord_num.append(sing_vals[0]/sing_vals[-1])
        # Wmats[-1] = Wmats[-1]/np.real(eigvals[0])
        proj_operators[count] = []
        for s in orient_const: #projecttions to top 3 eigenvectors
            proj_operators[count].append(np.random.randn(N_net,D))#np.real(s*eigvect[:,:D]+(1-s)*eigvect[:,-2*D:-1*D]))
        count += 1
    plt.plot(x,lower_coord+g*np.arange(1,N_net+1)**(-1-2/2),'r')   
    plt.plot(x,lower_coord+g*np.arange(1,N_net+1)**-1.0,'b')   
    # plt.ylim(10e-5,4)
    plt.xlabel('n',fontsize=14)
    plt.ylabel('$\lambda$',fontsize=14)
    plt.xscale('log')
    plt.yscale('log')
    # labellines.labelLines(plt.gca().get_lines(),zorder=2.5,fontsize=6)
    # plt.savefig('../Results/Figures/nspectrum.svg',transparent=True)
    plt.figure(dpi=300)
    
    eigarray = np.hstack(eiglist)
    plt.hist2d(eigarray[0],eigarray[1],100)
    #%% run simulations (example with Sphere)
    Networks = np.empty([n_mats,n_proj],dtype=np.object)
    T = 6#00 #the number of time points for which to simulate
    init_x = np.zeros(N_net)#np.random.randn(N_net)
    in_str = 1
    int_const = 1#0.01
    slope = 1
    
    PCAs = np.zeros([n_mats,n_proj,T,N_net])
    for w in range(n_mats):
        for p in range(n_proj):
            net_S2 = np.zeros([N_net,T,N**2])
            for i in range(N**2):    
                I = in_str*proj_operators[w][p]@S2[:,i]
                I = np.matlib.repmat(I,T,1).T 
                # I = I + 0.01*np.random.randn(N_net,T)
                I[:,1:] = 0
                net_S2[:,:,i] = low_rank_rnn(N_net,T,I,P=Wmats[w],init_x=init_x,mu=int_const,act_fun = lambda x: np.tanh(slope*x))
            # for t in range(T):
            #     PCAs[w,p,t,:] = PCA().fit(net_S2[:,t,:]).explained_variance_ratio_
            Networks[w,p] = net_S2
        print(w)
    
    
    #%%
    #Calculate topology
    Phom_object = Persistent_Homology()
    tstep = T-2#10
    times = np.arange(1,T,tstep)
    S2_rdmat = rankdata(pairwise_distances(S2.T))
    S2_hom = Phom_object.normalize(Phom_object.homology_analysis(S2.T,pairwise_distances,2,N**2)[1])#tda(S2.T,maxdim=2,n_perm=80)['dgms'])
    S2_H2 = np.max(S2_hom[2][:,1]-S2_hom[2][:,0])
    Net_homs = np.empty([n_mats,n_proj,len(times)],dtype=np.object)
    Net_rdmat = np.zeros([n_mats,n_proj,len(times)])
    S2_H2_net = np.zeros([n_mats,n_proj,len(times)])
    
    
    for t in range(len(times)):
        for i in range(n_mats):
            for j in range(n_proj):
                Net_homs[i,j,t] = Phom_object.normalize(Phom_object.homology_analysis(Networks[i,j][:,times[t],:].T,pairwise_distances,2,N**2)[1])
                Net_rdmat[i,j,t] = np.linalg.norm(S2_rdmat-rankdata(pairwise_distances(Networks[i,j][:,times[t],:].T)))/(len(S2_rdmat)**2)
                try:
                    S2_H2_net[i,j,t] = np.max(Net_homs[i,j,t][2][:,1]-Net_homs[i,j,t][2][:,0])
                except:
                    S2_H2_net[i,j,t] = 0
        print(t)
    S2_H2_net_.append(S2_H2_net)
S2_H2_net = np.mean(S2_H2_net_,0)    

#%%
X,Y = np.meshgrid(orient_const,decay_consts)
H2_destr = S2_H2-S2_H2_net[:,:,-1]
# np.save('Phase_map4.npy',H2_destr)
phase_boundary = decay_consts[np.argmin(np.abs(H2_destr-(np.mean(H2_destr,0))),0)]


plt.figure(dpi=300,figsize=(10,8))       
contours = plt.contour(X, Y, H2_destr, 1, colors='black')
plt.clabel(contours, inline=True, fontsize=8)
plt.contourf(X,Y,H2_destr,10,cmap='seismic',alpha=0.7)
# plt.hlines(g*1,0,1,'b')
# plt.hlines(g*2,0,1,'r')
# plt.imshow(H2_destr, extent=[0, 1, decay_consts[0], decay_consts[-1]], origin='lower',
#             cmap='RdYlGn_r', aspect='auto')
plt.clim(0,S2_H2)
plt.colorbar()
# plt.text(orient_const[6],decay_consts[-5],'Topology is lost',color='black',fontsize=8)
# plt.text(orient_const[-8],decay_consts[2],'Topology is preserved',color='black',fontsize=8)
plt.ylabel('s')#'Weight spectrum decay')        
plt.xlabel('Projection orientation')
plt.yscale('log')
plt.title('Topological destruction')
plt.tight_layout()
# plt.savefig('../Results/Figures/Phase_hom_orth.png',transparent=True)

plt.figure(dpi=300,figsize=(10,8))       
contours = plt.contour(X, Y, Net_rdmat[:,:,-1], 1, colors='black')
plt.clabel(contours, inline=True, fontsize=8)
plt.contourf(X,Y,Net_rdmat[:,:,-1],10,cmap='seismic',alpha=0.7)
# plt.imshow(H2_destr, extent=[0, 1, decay_consts[0], decay_consts[-1]], origin='lower',
#             cmap='RdYlGn_r', aspect='auto')
plt.colorbar()
# plt.clim(0,S2_H2)
# plt.text(orient_const[2],decay_consts[-3],'Topology is lost',color='black',fontsize=8)
# plt.text(orient_const[-8],decay_consts[15],'Topology is preserved',color='black',fontsize=8)
plt.ylabel('Weight spectrum decay')        
plt.xlabel('Projection orientation')
# plt.yscale('log')
plt.title('Topological disentanglement')
plt.tight_layout()

plt.figure(dpi=200)
plt.plot(H2_destr.flatten(),Net_rdmat[:,:,1].flatten(),'.')
print(spearmanr(H2_destr.flatten(),Net_rdmat[:,:,1].flatten()))

# dec_id, dec_id2 = -2, 4
# proj_id, proj_id2 = 3, -2
# plt.figure(dpi=300,figsize=(5,4))
# bottleneck_time(Net_homs[dec_id,proj_id,:],dim=2,plot=True)
# # plt.title('decay = '+str(round(decay_consts[dec_id],2))+', projection = ' + str(round(orient_const[proj_id],2)))
# plt.tight_layout()
# # plt.savefig('../Results/Figures/bd_1_exp.svg',transparent=True)
# plt.figure(dpi=300,figsize=(5,4))
# bottleneck_time(Net_homs[dec_id2,proj_id2,:],dim=2,plot=True)
# # plt.title('decay = '+str(round(decay_consts[dec_id2],2))+', projection = ' + str(round(orient_const[proj_id2],2)))
# plt.tight_layout()
# # plt.savefig('../Results/Figures/bd_2_exp.svg',transparent=True)

pca_obj1 = PCA(svd_solver = 'full')#n_components=3,svd_solver = 'full')
pca_obj2 = PCA(svd_solver = 'full')#n_components=3,svd_solver = 'full')
pca_id1 = pca_obj1.fit(Networks[dec_id,proj_id][:,-1].T)
pcat_id1 = pca_id1.transform(Networks[dec_id,proj_id][:,-1].T)
pca_id2 = pca_obj2.fit(Networks[dec_id2,proj_id2][:,-1].T)
pcat_id2 = pca_id2.transform(Networks[dec_id2,proj_id2][:,-1].T)

plt.figure(dpi=300,figsize=(5,4))
plt.plot(np.arange(1,N_net+1),pca_id1.explained_variance_ratio_,'k-o',linewidth=1)
plt.xscale('log')
plt.ylabel('Explained variance ratio',fontsize=14)
plt.xlabel('n',fontsize=14)
plt.ylim(0,1)
plt.xlim(1-0.06,N_net+1)
# plt.savefig('../Results/Figures/pca_dec_1.svg',transparent=True)


plt.figure(dpi=300,figsize=(5,4))
plt.plot(np.arange(1,N_net+1),pca_id2.explained_variance_ratio_,'k-o',linewidth=1)
plt.xscale('log')
plt.ylabel('Explained variance ratio',fontsize=14)
plt.xlabel('n',fontsize=14)
plt.ylim(0,1)
plt.xlim(1-0.06,N_net+1)
# plt.savefig('../Results/Figures/pca_dec_2.svg',transparent=True)

colmap = 'viridis'
xyz_lim1 = [np.min(pcat_id1),np.max(pcat_id1)]
plt.figure(dpi=300,figsize=(6,6))
ax1 = plt.axes(projection='3d')
ax1.scatter(pcat_id1[:,0],pcat_id1[:,1],pcat_id1[:,2],s=5,c=np.linspace(0,1,N**2),cmap=colmap,alpha=0.6)
ax1.axis('off')
ax1.set_xlim(xyz_lim1[0],xyz_lim1[1])
ax1.set_ylim(xyz_lim1[0],xyz_lim1[1])
ax1.set_zlim(xyz_lim1[0],xyz_lim1[1])
# plt.savefig('../Results/Figures/Pca_mfld_1.svg',transparent=True)


xyz_lim2 = [np.min(pcat_id2),np.max(pcat_id2)]
plt.figure(dpi=300,figsize=(6,6))
ax2 = plt.axes(projection='3d')
ax2.scatter(pcat_id2[:,0],pcat_id2[:,1],pcat_id2[:,2],s=5,c=np.linspace(0,1,N**2),cmap=colmap,alpha=0.6)
ax2.axis('off')
ax2.set_xlim(xyz_lim2[0],xyz_lim2[1])
ax2.set_ylim(xyz_lim2[0],xyz_lim2[1])
ax2.set_zlim(xyz_lim2[0],xyz_lim2[1])
# plt.savefig('../Results/Figures/Pca_mfld_2.svg',transparent=True)

#%%Study the impact of weight initialization on classifiers
from ML_models import *

S2_test = np.copy(S2)+.01*np.random.randn(len(S2),N**2)
n_bound = 15
bound_S2 = (np.random.randn(81,3)@mfld_gen.S2(n_bound,100)).T
# S2_test = S2_test/np.linalg.norm(S2_test,axis=0)
#define the classes on the manifold
nclass = 4
n_layers = 5
classes = np.linspace(0,N**2,nclass).astype(int)
labels = np.zeros(N**2)
labels_test = np.zeros(N**2)
count=0
for i in range(nclass-1):
    labels[classes[i]:classes[i+1]] = count
    labels_test[classes[i]:classes[i+1]] = count
    count += 1
activation_fields = {}
train_data = [torch.tensor(S2),torch.tensor(labels)]
test_data = [torch.tensor(S2_test),torch.tensor(labels_test)]
performance = np.zeros([n_mats,n_proj])
for m in range(n_mats):
    for p in range(n_proj):
        model = NN_class(N_net,D,nclass-1,torch.tensor(proj_operators[m][p]),torch.tensor(Wmats[m]),n_layers=n_layers).double()
        
        criterion1 = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        
        loss_curve = train(model,train_data,optimizer,criterion1,20)
        
        # plt.plot(loss_curve)
        
        predictions = predict(model,test_data[0].T)
        pca_pred = PCA(n_components=3).fit_transform(predictions[0])
        predicted_labels = np.argmax(predictions[1],1)
        performance[m,p] = np.sum(labels_test==predicted_labels)/len(labels)
        print(np.sum(labels_test==predicted_labels)/len(labels))
        print('Weight decay = '+str(m)+' ; Projection_orientation = '+str(p))
        
        activation_fields[str(m)] = []
        for i in range(n_layers):
            mod_act = model.lim_forward(test_data[0].T,i).detach().numpy()
            activation_fields[str(m)].append(mod_act.T)

        # plt.figure()
        # plt.plot(predicted_labels,'r')
        # plt.plot(labels,'k')

        fig = plt.figure(dpi=300,figsize=(5,2))
        
        # add subplot with projection='3d'
        ax1 = fig.add_subplot(131, projection='3d')        
        for i in range(nclass):
            ax1.scatter(S2_test[0,labels==i],S2_test[1,labels==i],S2_test[2,labels==i])
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_zticks([])
        
        ax2 = fig.add_subplot(132, projection='3d')                    
        for i in range(nclass):
            ax2.scatter(pca_pred[labels==i,0],pca_pred[labels==i,1],pca_pred[labels==i,2]) 
        ax2.set_xticks([])
        ax2.set_yticks([])
        # ax2.set_zticks([])

        ax3 = fig.add_subplot(133, projection='3d')                    
        for i in range(nclass):
            ax3.scatter(pca_pred[predicted_labels==i,0],pca_pred[predicted_labels==i,1],pca_pred[predicted_labels==i,2])
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.set_zticks([])
        plt.suptitle('Classification for s=' + str(round(decay_consts[m],2)))
        
        plt.tight_layout()
        # plt.savefig('../Figures/class_dec_'+str(m)+'.png',transparent=True)
        
        
X,Y = np.meshgrid(orient_const,decay_consts)
phase_boundary = decay_consts[np.argmin(performance,0)]


plt.figure(dpi=300,figsize=(10,8))       
contours = plt.contour(X, Y, performance, 1, colors='black')
plt.clabel(contours, inline=True, fontsize=8)
plt.contourf(X,Y,performance,10,cmap='seismic',alpha=0.7)
# plt.clim(0,1)
plt.colorbar()
plt.ylabel('Weight spectrum decay')        
plt.xlabel('Projection orientation')
# plt.yscale('log')
plt.title('Classifier performance')
plt.tight_layout()

#%%Comparison

plt.plot(H2_destr.flatten(),performance.flatten(),'.')
print(np.corrcoef(H2_destr.flatten(),performance.flatten()))



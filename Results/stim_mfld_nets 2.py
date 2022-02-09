import sys,os
sys.path.append(os.getcwd())
sys.path.append(os.getcwd()[:-7]+'Analysis')
from Algorithms import *

#%% Define input manifolds
N_S1 = 400 #the number of points from the manifold which to sample

#circle
S1 = gen_mfld(N_S1,'S1', 1)

#Sphere
N_S2 = 20
S2 = gen_mfld(N_S2,'S2', 1)

#Torus
N_T2 = 20
T2 = gen_mfld(N_T2, 'T2',1,0.66)

#%%
#define network parameters
N_net = 100 #the number of neurons in the network
T = 400 #the number of time points for which to simulate
init_x = np.zeros(N_net)#np.random.randn(N_net)
Wmat = 3.1*np.random.randn(N_net,N_net)#gen_weight_mat(N_net,rank=3,g=3,svd='qr_decomp',eigenvals=[3,3,3])[0]#
np.fill_diagonal(Wmat,0)
W_eig = np.linalg.eig(Wmat)[1]
in_str = 3.1
int_const = 0.01

#iterate over each stimulus

##Circle
S1_mix = np.random.randn(N_net,2)#np.real(W_eig[:,:2])#
net_S1 = np.zeros([N_net,T,N_S1])
for i in range(N_S1):
    I = in_str*S1_mix@S1[:,i]
    I = np.matlib.repmat(I,T,1).T    
    net_S1[:,:,i] = low_rank_rnn(N_net,T,I,P=Wmat,init_x=init_x,mu=int_const)

# plot_stimtime_funct(net_S1,5,5)

##Sphere
S2_mix = np.random.randn(N_net,3)#np.real(W_eig[:,:3])#
net_S2 = np.zeros([N_net,T,N_S2**2])
for i in range(N_S2**2):
    I = in_str*S2_mix@S2[:,i]
    I = np.matlib.repmat(I,T,1).T 
    net_S2[:,:,i] = low_rank_rnn(N_net,T,I,P=Wmat,init_x=init_x,mu=int_const)

# plot_stimtime_funct(net_S2,5,5)

##Torus
T2_mix = np.random.randn(N_net,3)#np.real(W_eig[:,:3])#
net_T2 = np.zeros([N_net,T,N_T2**2])
for i in range(N_T2**2):
    I = in_str*T2_mix@T2[:,i]
    I = np.matlib.repmat(I,T,1).T    
    net_T2[:,:,i] = low_rank_rnn(N_net,T,I,P=Wmat,init_x=init_x,mu=int_const)

# plot_stimtime_funct(net_T2,3,3)

#%%Manifold analysis with PCA
PCA_red = PCA()

#Analysis for circle
transformed_S1_net = np.zeros(np.shape(net_S1))
transformed_S2_net = np.zeros(np.shape(net_S2))
transformed_T2_net = np.zeros(np.shape(net_T2))
for i in range(T):
    transformed_S1_net[:,i,:] = PCA().fit_transform(net_S1[:,i,:].T).T
    transformed_S2_net[:,i,:] = PCA().fit_transform(net_S2[:,i,:].T).T
    transformed_T2_net[:,i,:] = PCA().fit_transform(net_T2[:,i,:].T).T


plot_points = [2,5,20,100]
fig = plt.figure(dpi=300)
for p in range(len(plot_points)):
    ax1 = plt.subplot(3,len(plot_points),p+1)
    ax1.scatter(transformed_S1_net[0,plot_points[p],:],transformed_S1_net[1,plot_points[p],:],c=np.linspace(0,1,N_S1))
    ax1.axis('off')
    ax2 = plt.subplot(3,len(plot_points),p+5,projection='3d')
    ax2.scatter(transformed_S2_net[0,plot_points[p],:],transformed_S2_net[1,plot_points[p],:],transformed_S2_net[2,plot_points[p],:],s=4,c=np.linspace(0,1,N_S2**2))
    ax2.axis('off')
    ax3 = plt.subplot(3,len(plot_points),p+9,projection='3d')
    ax3.scatter(transformed_T2_net[0,plot_points[p],:],transformed_T2_net[1,plot_points[p],:],transformed_T2_net[2,plot_points[p],:],s=4,c=np.linspace(0,1,N_T2**2))
    ax3.axis('off')
plt.tight_layout()

#%%animate pca dynamics
from matplotlib import animation

#circle
def animate(i,embedding,plot_basis = 'off', basis_vects=0, stim_vects=0):
    ax.cla()
    ax.scatter(embedding[0,i,:],embedding[1,i,:],embedding[2,i,:],s=4,c=np.linspace(0,1,len(embedding[0,0,:])))
    if plot_basis=='on':
        X,Y,Z = np.zeros(3), np.zeros(3), np.zeros(3) 
        U, V, W = basis_vects[0], basis_vects[1], basis_vects[2]
        US, VS, WS = stim_vects[0], stim_vects[1], stim_vects[2]
        ax.quiver(X,Y,Z,U,V,W,color='k')
        ax.quiver(X,Y,Z,US,VS,WS,color='g')
    x = np.mean(embedding[0,i,0])
    y = np.mean(embedding[1,i,1])
    z = np.mean(embedding[2,i,2])
    inclin = np.arctan(np.sqrt(x**2+y**2)/z)
    if x>0:
        azim = np.arctan(y/x)
    elif x<0:
        azim = np.arctan(y/x)+np.pi
    else:
        azim = np.pi/2    
    # ax.set_xlim3d(-1,1)   
    # ax.set_ylim3d(-1,1)   
    # ax.set_zlim3d(-1,1)
    # ax.view_init(inclin,azim)
    # ax.axis('off')

#%%Circle animation
fig = plt.figure(figsize=(6,6),dpi=200,constrained_layout=True)
ax = plt.subplot(111,projection='3d')
ax.axis('off')
anim_S1 = animation.FuncAnimation(fig, animate, blit=False, frames = T,fargs=(transformed_S1_net,),interval=300)
# anim_S2 = animation.FuncAnimation(fig, animate, blit=False, frames = T,fargs=(net_S1,'on',Wmat, S1_mix,),interval=300)

#%%Sphere animation
fig = plt.figure(figsize=(6,6),dpi=200,constrained_layout=True)
ax = plt.subplot(111,projection='3d')
ax.axis('off')
anim_S2 = animation.FuncAnimation(fig, animate, blit=False, frames = T,fargs=(transformed_S2_net,),interval=300)
# anim_S2 = animation.FuncAnimation(fig, animate, blit=False, frames = T,fargs=(net_S2,'on',Wmat, S2_mix,),interval=300)

#%%Torus animation
fig = plt.figure(figsize=(6,6),dpi=200,constrained_layout=True)
ax = plt.subplot(111,projection='3d')
ax.axis('off')
anim_T2 = animation.FuncAnimation(fig, animate, blit=False, frames = T,fargs=(transformed_T2_net,),interval=300)
# anim_T2 = animation.FuncAnimation(fig, animate, blit=False, frames = T,fargs=(net_T2,'on',Wmat, T2_mix,),interval=300)

#%%Topological analysis of the manifolds through time
phoms_S1 = []
phoms_S2 = []
phoms_T2 = []

tpoints = 300
top_points = np.arange(1,tpoints)
top_method = 'LP'
for i in top_points:#range(T):
    phoms_S1.append(normal_bd_dist(full_hom_analysis(net_S1[:,i,:].T,metric=top_method,dim=2,perm=100,R=-1,Eps=.1)[1]))#tda(net_S1[:,i,:].T,maxdim=2)['dgms'])#
    phoms_S2.append(normal_bd_dist(full_hom_analysis(net_S2[:,i,:].T,metric=top_method,dim=2,perm=100,R=-1,Eps=.1)[1]))#tda(net_S2[:,i,:].T,maxdim=2)['dgms'])#
    phoms_T2.append(normal_bd_dist(full_hom_analysis(net_T2[:,i,:].T,metric=top_method,dim=2,perm=100,R=-1,Eps=.1)[1]))#tda(net_T2[:,i,:].T,maxdim=2)['dgms'])#
bcurves_S1 = extract_pers(phoms_S1,[1,1,1])
bcurves_S2 = extract_pers(phoms_S2,[1,1,1])
bcurves_T2 = extract_pers(phoms_T2,[1,2,1])   

clrs = ['b','r','g']
plt.figure(dpi=300)
for i in range(3):
    plt.subplot(3,1,1)
    plt.plot(bcurves_S1[i],clrs[i])
    plt.subplot(3,1,2)
    plt.plot(bcurves_S2[i],clrs[i])
    plt.subplot(3,1,3)
    plt.plot(bcurves_T2[i],clrs[i])
plt.tight_layout()

#%% Calculate bottleneck distances between diagrams
S1_H1_dmat = bottleneck_dmat(phoms_S1,phoms_S1)
S2_H1_dmat = bottleneck_dmat(phoms_S2,phoms_S2)
S2_H2_dmat = bottleneck_dmat(phoms_S2,phoms_S2,dim=2)
T2_H1_dmat = bottleneck_dmat(phoms_T2,phoms_T2)
T2_H2_dmat = bottleneck_dmat(phoms_T2,phoms_T2,dim=2)

#%% plot topological changes
#Prepare matching between first and last step
_,S1_matching = persim.bottleneck(phoms_S1[0][1],phoms_S1[-1][1],matching=True)
_,S2_H1_matching = persim.bottleneck(phoms_S2[0][1],phoms_S2[-1][1],matching=True)
_,S2_H2_matching = persim.bottleneck(phoms_S2[0][2],phoms_S2[-1][2],matching=True)
_,T2_H1_matching = persim.bottleneck(phoms_T2[0][1],phoms_T2[-1][1],matching=True)
_,T2_H2_matching = persim.bottleneck(phoms_T2[0][2],phoms_T2[-1][2],matching=True)

plt.figure(dpi=200,constrained_layout=False)

plt.subplot(5,2,1)
persim.bottleneck_matching(phoms_S1[0][1], phoms_S1[-1][1], S1_matching)
plt.subplot(5,2,2)
plt.imshow(S1_H1_dmat)
plt.subplot(5,2,3)
persim.bottleneck_matching(phoms_S2[0][1], phoms_S2[-1][1], S2_H1_matching)
plt.subplot(5,2,4)
plt.imshow(S2_H1_dmat)
plt.subplot(5,2,5)
persim.bottleneck_matching(phoms_S2[0][2], phoms_S2[-1][2], S2_H2_matching)
plt.subplot(5,2,6)
plt.imshow(S2_H1_dmat)
plt.subplot(5,2,7)
persim.bottleneck_matching(phoms_T2[0][1], phoms_T2[-1][1], T2_H1_matching)
plt.subplot(5,2,8)
plt.imshow(T2_H2_dmat)
plt.subplot(5,2,9)
persim.bottleneck_matching(phoms_T2[0][2], phoms_T2[-1][2], T2_H2_matching)
plt.subplot(5,2,10)
plt.imshow(T2_H2_dmat)

plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[],xlabel='',ylabel='')
plt.tight_layout()


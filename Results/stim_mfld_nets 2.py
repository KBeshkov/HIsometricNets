import sys,os
sys.path.append(os.getcwd())
sys.path.append(os.getcwd()[:-7]+'Analysis')
from Algorithms import *
import numpy as np

#%% Define input manifolds
N_S1 = 1600 #the number of points from the manifold which to sample

#circle
theta_S1 = np.linspace(0,2*np.pi,N_S1)
S1 = np.array([np.cos(theta_S1),np.sin(theta_S1)])

#Sphere
N_S2 = 40
phi = np.linspace(0,np.pi,N_S2)
theta = np.linspace(0,2*np.pi,N_S2)
Phi, Theta = np.meshgrid(phi,theta)
Phi_S2, Theta_S2 = Phi.flatten(), Theta.flatten()
S2 = np.array([np.cos(Theta_S2)*np.sin(Phi_S2),np.sin(Theta_S2)*np.sin(Phi_S2),np.cos(Phi_S2)])

#Torus
N_T2 = 40
R = 1
r= 0.66
phi = np.linspace(0,2*np.pi,N_T2)
theta = np.linspace(0,2*np.pi,N_T2)
Phi, Theta = np.meshgrid(phi,theta)
Phi_T2, Theta_T2 = Phi.flatten(), Theta.flatten()
T2 = np.array([(R+r*np.cos(Theta_T2))*np.cos(Phi_T2),(R+r*np.cos(Theta_T2))*np.sin(Phi_T2),r*np.sin(Theta_T2)])
#%%
#define network parameters
N_net = 100 #the number of neurons in the network
T = 400 #the number of time points for which to simulate
init_x = np.zeros(N_net)#np.random.randn(N_net)
Wmat = 15.1*np.random.randn(N_net,N_net)#gen_weight_mat(N_net,rank=3,g=3,svd='qr_decomp',eigenvals=[3,3,3])[0]#
np.fill_diagonal(Wmat,0)
W_eig = np.linalg.eig(Wmat)[1]
in_str = 1
int_const = 0.01

#iterate over each stimulus

##Circle
S1_mix = np.random.randn(N_net,2)#np.real(W_eig[:,:2])#
net_S1 = np.zeros([N_net,T,N_S1])
for i in range(N_S1):
    I = in_str*S1_mix@S1[:,i]
    I = np.matlib.repmat(I,T,1).T    
    net_S1[:,:,i] = low_rank_rnn(N_net,T,I,P=Wmat,init_x=init_x,mu=int_const)

plot_stimtime_funct(net_S1,5,5)

##Sphere
S2_mix = np.random.randn(N_net,3)#np.real(W_eig[:,:3])#
net_S2 = np.zeros([N_net,T,N_S2**2])
for i in range(N_S2**2):
    I = in_str*S2_mix@S2[:,i]
    I = np.matlib.repmat(I,T,1).T 
    net_S2[:,:,i] = low_rank_rnn(N_net,T,I,P=Wmat,init_x=init_x,mu=int_const)

plot_stimtime_funct(net_S2,5,5)

##Torus
T2_mix = np.random.randn(N_net,3)#np.real(W_eig[:,:3])#
net_T2 = np.zeros([N_net,T,N_T2**2])
for i in range(N_T2**2):
    I = in_str*T2_mix@T2[:,i]
    I = np.matlib.repmat(I,T,1).T    
    net_T2[:,:,i] = low_rank_rnn(N_net,T,I,P=Wmat,init_x=init_x,mu=int_const)

plot_stimtime_funct(net_T2,3,3)

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
def animate(i,embedding):
    ax.cla()
    ax.scatter(embedding[0,i,:],embedding[1,i,:],embedding[2,i,:],s=4,c=np.linspace(0,1,len(embedding[0,0,:])))
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
    ax.view_init(inclin,azim)
    ax.axis('off')

#%%Circle animation
fig = plt.figure(figsize=(6,6),dpi=200,constrained_layout=True)
ax = plt.subplot(111,projection='3d')
ax.axis('off')
anim_S1 = animation.FuncAnimation(fig, animate, blit=False, frames = T,fargs=(transformed_S1_net,),interval=300)

#anim_S1.save(os.getcwd()[:-7]+'/Figures//S1_animation.gif', writer='Pillow')
#%%Sphere animation
fig = plt.figure(figsize=(6,6),dpi=200,constrained_layout=True)
ax = plt.subplot(111,projection='3d')
ax.axis('off')
anim_S2 = animation.FuncAnimation(fig, animate, blit=False, frames = T,fargs=(transformed_S2_net,),interval=300)

#anim_S2.save(os.getcwd()[:-7]+'/Figures//S2_animation.gif', writer='Pillow')
#%%Torus animation
fig = plt.figure(figsize=(6,6),dpi=200,constrained_layout=True)
ax = plt.subplot(111,projection='3d')
ax.axis('off')
anim_T2 = animation.FuncAnimation(fig, animate, blit=False, frames = T,fargs=(transformed_T2_net,),interval=300)

#anim_T2.save(os.getcwd()[:-7]+'\Figures\\T2_animation.gif', writer='Pillow')
#%%Topological analysis of the manifolds through time
phoms_S1 = []
phoms_S2 = []
phoms_T2 = []

tpoints = 200
top_points = np.arange(1,tpoints)
top_method = 'LP'
for i in top_points:#range(T):
    phoms_S1.append(normal_bd_dist(full_hom_analysis(net_S1[:,i,:].T,metric=top_method,dim=2,perm=300,R=-1,Eps=.1)[1]))#tda(net_S1[:,i,:].T,maxdim=2)['dgms'])#
    phoms_S2.append(normal_bd_dist(full_hom_analysis(net_S2[:,i,:].T,metric=top_method,dim=2,perm=300,R=-1,Eps=.1)[1]))#tda(net_S2[:,i,:].T,maxdim=2)['dgms'])#
    phoms_T2.append(normal_bd_dist(full_hom_analysis(net_T2[:,i,:].T,metric=top_method,dim=2,perm=300,R=-1,Eps=.1)[1]))#tda(net_T2[:,i,:].T,maxdim=2)['dgms'])#
bcurves_S1 = extract_pers(phoms_S1,[1,1,1])
bcurves_S2 = extract_pers(phoms_S2,[1,1,1])
bcurves_T2 = extract_pers(phoms_T2,[1,2,1])   


# fig = plt.figure(dpi=300)
# for p in range(len(top_points)):
#     ax1 = plt.subplot(3,len(top_points),p+1)
#     plot_diagrams(phoms_S1[p],legend='off')
#     ax1.axis('off')
#     ax2 = plt.subplot(3,len(top_points),p+tpoints+1)
#     plot_diagrams(phoms_S2[p],legend='off')
#     ax2.axis('off')
#     ax3 = plt.subplot(3,len(top_points),p+2*tpoints+1)
#     plot_diagrams(phoms_T2[p],legend='off')    
#     ax3.axis('off')
# plt.tight_layout()

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
#%%fit stimulusXtime tunning functions for each manifold
N_rbf = 300

times = np.arange(0,int_const*T,int_const)

S1_space = np.meshgrid(times,theta_S1)
S1_space = np.vstack([S1_space[0].flatten(),S1_space[1].flatten()]).T
T_centers_S1 = np.max(times)*np.random.rand(N_rbf)
Theta_centers_S1 = np.max(theta_S1)*np.random.rand(N_rbf)
rbf_centers_S1 = np.vstack([T_centers_S1,Theta_centers_S1]).T

S1_basis_pred = np.zeros(np.shape(net_S1))
kern_sizes = np.linspace(0,10,N_rbf)
for i in range(N_net):
    S1_rbf_kerns = RBF_regress(net_S1[i].flatten(),S1_space,rbf_centers_S1,kern_sizes,weight=np.array([1,0.01]))
    S1_basis_pred[i] = (S1_rbf_kerns[1]@S1_rbf_kerns[0]).reshape([T,N_S1])
    print(i)

plt.figure(dpi=300)
plt.subplot(2,1,1)
plt.imshow(net_S1[-1])
plt.subplot(2,1,2)
plt.imshow(S1_basis_pred[-1])

#%%
#metric tensor
dXdTheta = -2*S1_basis_pred*(S1_space-rbf_centers)**2
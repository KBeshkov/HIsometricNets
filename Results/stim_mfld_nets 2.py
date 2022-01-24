from Algorithms import *
import numpy as np

#%% Define input manifolds
N_S1 = 900 #the number of points from the manifold which to sample

#circle
theta = np.linspace(0,2*np.pi,N_S1)
S1 = np.array([np.cos(theta),np.sin(theta)])+2

#Sphere
N_S2 = 30
phi = np.linspace(0,np.pi,N_S2)
theta = np.linspace(0,2*np.pi,N_S2)
Phi, Theta = np.meshgrid(phi,theta)
Phi, Theta = Phi.flatten(), Theta.flatten()
S2 = np.array([np.cos(Theta)*np.sin(Phi),np.sin(Theta)*np.sin(Phi),np.cos(Phi)])+2

#Torus
N_T2 = 30
R = 1
r= 0.66
phi = np.linspace(0,2*np.pi,N_T2)
theta = np.linspace(0,2*np.pi,N_T2)
Phi, Theta = np.meshgrid(phi,theta)
Phi, Theta = Phi.flatten(), Theta.flatten()
T2 = np.array([(R+r*np.cos(Theta))*np.cos(Phi),(R+r*np.cos(Theta))*np.sin(Phi),r*np.sin(Theta)])+2
#%%
#define network parameters
N_net = 100 #the number of neurons in the network
T = 1000 #the number of time points for which to simulate
init_x = np.random.randn(N_net)
Wmat = 15.1*np.random.randn(N_net,N_net)#gen_weight_mat(N_net,rank=3,g=3,svd='qr_decomp',eigenvals=[3,3,3])[0]#
np.fill_diagonal(Wmat,0)
W_eig = np.linalg.eig(Wmat)[1]
in_str = 1

#iterate over each stimulus

##Circle
S1_mix = np.random.randn(N_net,2)#np.real(W_eig[:,:2])#
net_S1 = np.zeros([N_net,T,N_S1])
for i in range(N_S1):
    I = in_str*S1_mix@S1[:,i]
    I = np.matlib.repmat(I,T,1).T    
    net_S1[:,:,i] = low_rank_rnn(N_net,T,I,P=Wmat,init_x=init_x)

plot_stimtime_funct(net_S1,5,5)

##Sphere
S2_mix = np.random.randn(N_net,3)#np.real(W_eig[:,:3])#
net_S2 = np.zeros([N_net,T,N_S2**2])
for i in range(N_S2**2):
    I = in_str*S2_mix@S2[:,i]
    I = np.matlib.repmat(I,T,1).T 
    net_S2[:,:,i] = low_rank_rnn(N_net,T,I,P=Wmat,init_x=init_x)

plot_stimtime_funct(net_S2,5,5)

##Torus
T2_mix = np.random.randn(N_net,3)#np.real(W_eig[:,:3])#
net_T2 = np.zeros([N_net,T,N_T2**2])
for i in range(N_T2**2):
    I = in_str*T2_mix@T2[:,i]
    I = np.matlib.repmat(I,T,1).T    
    net_T2[:,:,i] = low_rank_rnn(N_net,T,I,P=Wmat,init_x=init_x)

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


plot_points = [2,3,41,55]
fig = plt.figure(dpi=200)
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
    ax.view_init(embedding[0,i,0]-embedding[0,i+1,0],embedding[1,i,0]-embedding[1,i+1,0])
    ax.axis('off')

#%%Circle animation
fig = plt.figure(figsize=(6,6),dpi=200,constrained_layout=True)
ax = plt.subplot(111,projection='3d')
ax.axis('off')
anim_S1 = animation.FuncAnimation(fig, animate, blit=False, frames = T,fargs=(transformed_S1_net,),interval=200)

#%%Sphere animation
fig = plt.figure(figsize=(6,6),dpi=200,constrained_layout=True)
ax = plt.subplot(111,projection='3d')
ax.axis('off')
anim_S2 = animation.FuncAnimation(fig, animate, blit=False, frames = T,fargs=(transformed_S2_net,),interval=200)

#%%Torus animation
fig = plt.figure(figsize=(6,6),dpi=200,constrained_layout=True)
ax = plt.subplot(111,projection='3d')
ax.axis('off')
anim_T2 = animation.FuncAnimation(fig, animate, blit=False, frames = T,fargs=(transformed_T2_net,),interval=200)

#%%Topological analysis of the manifolds through time
phoms_S1 = []
phoms_S2 = []
phoms_T2 = []

for i in [5,10,50,500]:#range(T):
    phoms_S1.append(full_hom_analysis(net_S1[:,i,:].T,metric='geodesic',dim=2,perm=300,R=-1,Eps=1000)[1])#tda(net_S1[:,i,:].T,maxdim=2)['dgms'])#
    phoms_S2.append(full_hom_analysis(net_S2[:,i,:].T,metric='geodesic',dim=2,perm=300,R=-1,Eps=1000)[1])#tda(net_S2[:,i,:].T,maxdim=2)['dgms'])#
    phoms_T2.append(full_hom_analysis(net_T2[:,i,:].T,metric='geodesic',dim=2,perm=300,R=-1,Eps=1000)[1])#tda(net_T2[:,i,:].T,maxdim=2)['dgms'])#
    
plot_points = [2,5,10,50]
fig = plt.figure(dpi=200)
for p in range(len(plot_points)):
    ax1 = plt.subplot(3,len(plot_points),p+1)
    plot_diagrams(phoms_S1[p],legend='off')
    ax2 = plt.subplot(3,len(plot_points),p+5)
    plot_diagrams(phoms_S2[p],legend='off')
    ax3 = plt.subplot(3,len(plot_points),p+9)
    plot_diagrams(phoms_T2[p],legend='off')    
plt.tight_layout()
#%%fit stimulusXtime tunning functions for each manifold
times = np.arange(0,0.05*T,0.05)
T2_time_coords = np.hstack([T2,times])

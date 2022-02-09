#phase diagram of 
import sys,os
sys.path.append(os.getcwd())
sys.path.append(os.getcwd()[:-7]+'Analysis')
from Algorithms import *

#%% generate manifolds
N = 10
S1 = gen_mfld(N**2,'S1', 1)
S2 = gen_mfld(N,'S2', 1)
T2 = gen_mfld(N, 'T2',1,0.66)

#%%Generate weight matrices with a given spectrum
N_net = 80
D = 3
n_mats = 40
n_proj = 18

Wmats = []
decay_consts = np.logspace(-1.4, -0.4, n_mats)
g = 3
proj_operators = [0]*n_mats
orient_const = np.linspace(0,1,n_proj)

count = 0
for i in decay_consts:
    eigvals = g*(np.exp(-i*np.linspace(0,N_net,N_net)))
    Wmats.append(gen_weight_mat(N_net,N_net,g=0,svd='qr_decomp',eigenvals=eigvals)[0])
    # Wmats.append(gen_weight_mat(N_net,count+1,g=1,svd='qr_decomp',eigenvals=g*np.ones(count+1))[0])
    eigval, eigvect = np.linalg.eig(Wmats[-1])
    proj_operators[count] = []
    for s in orient_const: #projecttions to top 3 eigenvectors
        proj_operators[count].append(np.real(s*eigvect[:,:D]+(1-s)*eigvect[:,10*D:11*D]))
    count += 1
    
#%% run simulations (example with Sphere)
Networks = np.empty([n_mats,n_proj],dtype=np.object)
T = 400 #the number of time points for which to simulate
init_x = np.zeros(N_net)#np.random.randn(N_net)
in_str = 3
int_const = 0.01
slope = 1

for w in range(n_mats):
    for p in range(n_proj):
        net_S2 = np.zeros([N_net,T,N**2])
        for i in range(N**2):    
            I = in_str*proj_operators[w][p]@S2[:,i]
            I = np.matlib.repmat(I,T,1).T 
            net_S2[:,:,i] = low_rank_rnn(N_net,T,I,P=Wmats[w],init_x=init_x,mu=int_const,act_fun = lambda x: np.tanh(slope*x))
        Networks[w,p] = net_S2
    print(w)

#%%
#Calculate topology
S2_hom = normal_bd_dist(tda(S2.T,maxdim=2,n_perm=90)['dgms'])
S2_H2 = np.max(S2_hom[2][:,1]-S2_hom[2][:,0])
Net_homs = np.empty([n_mats,n_proj],dtype=np.object)
S2_H2_net = np.zeros([n_mats,n_proj])

for t in range(T-1,T):
    for i in range(n_mats):
        for j in range(n_proj):
            Net_homs[i,j] = normal_bd_dist(tda(Networks[i,j][:,t,:].T,maxdim=2,n_perm=80)['dgms'])
            try:
                S2_H2_net[i,j] = np.max(Net_homs[i,j][2][:,1]-Net_homs[i,j][2][:,0])
            except:
                S2_H2_net[i,j] = 0
    print(t)


#%%
X,Y = np.meshgrid(orient_const,decay_consts)
H2_destr = S2_H2-S2_H2_net
phase_boundary = decay_consts[np.argmin(np.abs(H2_destr-(np.mean(H2_destr,0))),0)]


plt.figure(dpi=200)       
plt.contourf(X,Y,H2_destr,250,cmap='RdYlGn_r')
plt.colorbar()
plt.text(orient_const[2],decay_consts[-2],'Topology is lost',color='navy')
plt.text(orient_const[-8],decay_consts[9],'Topology is preserved',color='navy')
plt.ylabel('Weight spectrum decay')        
plt.xlabel('Projection orientation')
# plt.yscale('log')
plt.title('Topology destruction')


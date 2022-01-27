#vfield approximation - load data here
int_const = 0.001
angl_const = theta_S1[1]-theta_S1[0]
snd_int = int_const**2
#%%Change of activity with respect to a stimulus

S1_deriv = net_S1[:,:,1:]-net_S1[:,:,:-1]
gS1 = np.zeros(np.shape(net_S1))
times = np.arange(0,T*int_const,int_const)

dS1ds = np.array([-np.sin(theta_S1),np.cos(theta_S1)])
for t in range(T):
    gS1[:,t,0] = net_S1[:,t,0]
    for a in range(N_S1-1):
        # ds = ((np.eye(N_net)-int_const*np.eye(N_net)+int_const*Wmat)**times[t])
        # ds[np.isnan(ds)]=0
        gS1[:,t+1,a+1] =  gS1[:,t+1,a]+angl_const*NN_shift_operator(int_const*S1_mix@dS1ds[:,a],Wmat,t,int_const,S1_mix,dS1ds[:,a])+angl_const*int_const*S1_mix@dS1ds[:,a]#net_S1[:,t,a]+angl_const*(S1_mix@dS1ds[:,a])
        
plt.plot(gS1[23,32,:])
plt.plot(net_S1[23,32,:])

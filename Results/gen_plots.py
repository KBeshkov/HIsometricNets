import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
path = '../Data/FGSM/'
rocket = sns.color_palette("rocket",10)

epsilons=np.logspace(-2,0,10)
class_curves = {}
distance_curves = {}
for filename in os.listdir(path):
    if filename[:5]=='class':
        print(filename)
        class_curves[int(filename[17:-4])]=np.load(path+filename)
    # elif filename[:9]=='distances':
        # class_curves[int(filename[14:-4])]=np.load(path+filename)
        
curve_mat = np.vstack(class_curves.values())


legend_lbls = [f'$\\beta = 10^{i}$' for i in range(-1,len(class_curves))]
legend_lbls[0]=0

plt.figure()
for count,i in enumerate(np.sort(list(class_curves.keys()))):
    plt.plot(epsilons,class_curves[i],'-o',color=rocket[count])
plt.xscale('log')
plt.grid('on')
plt.legend(legend_lbls)
plt.xlabel('$\epsilon$')
plt.ylabel('Classification rate')
plt.ylim(0,1)
plt.savefig('../Figures/Classification_robustness.png',dpi=1000)
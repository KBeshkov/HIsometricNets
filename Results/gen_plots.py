import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
plt.rcParams.update({'font.size': 14})
rocket = sns.color_palette("rocket",9)

attack = 'fgsm_temp/'
path = '../Data/'+attack

epsilons=np.logspace(-2,0,10)
class_curves = {}
distance_curves = {}
for filename in os.listdir(path):
    if filename[:5]=='class':
        print(filename)
        class_curves[filename[17:-4]]=np.load(path+filename)
        
curve_mat = np.vstack(class_curves.values())


legend_lbls = []

plt.figure()
for count,i in enumerate(np.sort(list(class_curves.keys()))):
    plt.plot(epsilons,class_curves[i],'-o',color=rocket[count],linewidth=3)
    legend_lbls.append('$\\beta = $'+str(i))
legend_lbls[0]='CSE'
plt.xscale('log')
plt.grid('on')
plt.legend(legend_lbls)
plt.xlabel('$\epsilon$')
plt.ylabel('Classification rate')
plt.ylim(0,1)
plt.tight_layout()
# plt.savefig('../Figures/Classification_robustness_fgsm.png',dpi=1000,transparent=True)

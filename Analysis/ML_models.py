#NN classifier
from Algorithms import *
from metrics import *
import os
import torch
from torch import nn
import torch.optim as optim
import numpy as np
import scipy.stats as st
#from jacobian import JacobianReg
from torch.autograd import Variable
#from torchviz import make_dot
from torch.utils.data import Dataset
import torch.nn.functional as FF
from torchvision import datasets
from torchvision.transforms import ToTensor, Resize, Grayscale, Normalize
from torchvision.io import read_image
from scipy import special
from scipy.sparse.csgraph import shortest_path
plt.style.use('default')#../misc/report.mplstyle')

class NN_class(nn.Module):
    def __init__(self,n_neurons,n_inputs,n_classes,projection,weights,n_layers=1):
        super(NN_class, self).__init__()
        self.n_neurons = n_neurons
        self.n_inputs = n_inputs
        self.n_classes = n_classes
        self.projection = projection
        self.weights = weights
        self.n_layers = n_layers
        
        self.layers = nn.ModuleList() #[]
        self.inp_layer = nn.Linear(n_inputs,n_neurons,bias=True)
        if len(self.projection)>0:
            self.inp_layer.weight.data = projection
        self.act_fun = nn.Tanh()#torch.sin##nn.Hardtanh()#
        for i in range(n_layers):
            out_layer = nn.Linear(n_neurons,n_neurons,bias=True)
            # c = torch.tensor(6/self.n_neurons)
            # torch.nn.init.normal_(out_layer.weight, mean=0.0, std=1/self.n_neurons)#torch.nn.init.uniform_(out_layer.weight,a=-30*torch.sqrt(c),b=30*torch.sqrt(c))#
            if len(self.weights)>0:
                out_layer.weight.data = weights
            self.layers.append(out_layer.double())
        self.class_layer = nn.Linear(n_neurons,n_classes)

        
    def forward(self,x):
        out1 = self.act_fun(self.inp_layer(x.double().T))
        for layer in self.layers:
            out1 = self.act_fun(layer(out1))
        out2 = self.class_layer(out1)
        return out1, out2

    
    def lim_forward(self,x,layer_stop):
        out1 = self.act_fun(self.inp_layer(x.double()))
        for layer in self.layers[:-layer_stop]:
            out1 = self.act_fun(layer(out1))
        return out1
    
def assign_points(mod_points,mfld_points,asgn_type='rand'):
    if asgn_type=='rand':
        assg = torch.tensor(np.random.choice(np.arange(0,len(mod_points)),size=len(mfld_points)))
    elif asgn_type=='dist':
        dist = torch.cdist(mod_points,mfld_points.float())
        assg = torch.argmin(dist,1)
    elif asgn_type=='pairing':
        dist = torch.cdist(mod_points,mfld_points.float())
        sort_dist = torch.min(dist,1).values
        sort_ids = torch.sort(sort_dist).indices
        assg = torch.zeros(len(dist.T))
        for s in range(len(sort_ids)):
            assg[s] = torch.argmin(dist[:,sort_ids[s]])
            dist[assg[s].long(),:] = torch.inf
    return assg.long()

class TopologicalDestroyerLoss(nn.Module):
    def __init__(self,model, lambda1, lambda2,gluer_points,cutter_points):
        super().__init__()
        self.model = model
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.gluers = gluer_points
        self.cutters = cutter_points
        
    def clossest_point(self,point):
        cl_gluer = self.gluers[torch.argmin(torch.cdist(point,self.gluers))]
        cl_cutter = self.cutters[torch.argmin(torch.cdist(point,self.cutters))]
        return cl_gluer, cl_cutter
        
    def Loss(self,y1,yinds=None,point_type = 'closest',assigned_points=None):
        if point_type == 'closest':
            neighbors = self.clossest_point(y1)
            gluer_point = neighbors[0]
            cutter_point = neighbors[1]
        elif point_type == 'assigned':
            gluer_point = self.gluers[assigned_points][yinds]
            cutter_point = self.cutters[assigned_points][yinds]
        crit1 = nn.MSELoss()
        crit2 = nn.MSELoss()
        glue_loss = self.lambda1*crit1(gluer_point,y1)#torch.dot(self.lambdas1,torch.tensor([torch.trace(torch.cdist(model_gluers,y1))]))
        cut_loss = -self.lambda2*crit2(cutter_point,y1)#torch.dot(self.lambdas2,torch.tensor([(torch.trace(torch.cdist(model_cutters,y1)))]))
        loss = glue_loss+cut_loss
        return loss

class ManifoldLagrangeLoss(nn.Module):
    def __init__(self, lambda1,mfld_type):
        super().__init__()
        self.lambda1 = lambda1
        self.mfld_type = mfld_type
        
    def point_projection(self,point):
        if self.mfld_type == 'Sn':
            return point/torch.norm(point)
        if self.mfld_type == 'T2':
            r = 6.6
            R = 10
            x,y,z = point.T
            lambd = ((2*r**2+z)+torch.sqrt(4*r**4-8*z*r**2+z**2))/(2*r**2)
            w1 = y/(1-lambd)-(lambd*R)/(torch.sqrt(r**2-z**2)+(1-lambd)*R)
            w2 = x/(1-lambd)-(lambd*R)/(torch.sqrt(r**2-z**2)+(1-lambd)*R)
            w3 = z/(1-lambd)
            return torch.tensor([w1,w2,w3])
        
    def Loss(self,y1,yinds,*args):
        ws = []
        for i in range(len(yinds)):
            ws.append(self.point_projection(y1[i]))
        ws = torch.vstack(ws)
        crit = nn.MSELoss()
        loss = self.lambda1*crit(ws,y1)
        return loss
        

class MetricPreservationLoss(nn.Module):
    def __init__(self, lambda1, lambda2):
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        
    def Loss(self,y1,y1_dmats,Y,metric=torch.cdist):
        est_dmats = metric(y1,y1)
        crit1 = nn.MSELoss()
        loss = self.lambda1*crit1(Y*est_dmats/(self.lambda2*torch.max(Y*est_dmats)),Y*y1_dmats/torch.max(Y*y1_dmats))**2
        return loss, (torch.max(Y*est_dmats)/torch.max(Y*y1_dmats)).item()

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=10, lambda1=1, lambda2=1):
        self.margin = margin
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        
    def Loss(self,y1,Y):
        D = torch.cdist(y1,y1)
        crit1 = nn.MSELoss()
        crit2 = nn.MSELoss()
        loss = self.lambda1*crit1(Y*D,torch.zeros((D).shape,dtype=torch.double))
        +self.lambda2*crit2(nn.ReLU()((1-Y)*(self.margin-D)**2),torch.zeros((D).shape,dtype=torch.double))
        return loss
  
def pair_labels(labels):
    Labels_D = torch.zeros([len(labels),len(labels)])
    for l in range(len(labels)):
        for k in range(len(labels)):
            if labels[l]==labels[k]:
                Labels_D[l,k] = 1
    return Labels_D
        
def train(net,train_dat,optimizer,criterion_weights,dmat,epochs,batch_sz=20):
    running_loss = 0.0
    loss_curve = []
    loss1_curve = []
    loss2_curve = []
    batch_ind = torch.arange(0,len(train_dat[1].T))#torch.tensor(np.random.choice(np.arange(0,len(train_dat[1].T)),batch_sz,replace=False))
    inputs = torch.tensor(train_dat[0][:,batch_ind])
    inputs.requires_grad = True
    labels = torch.tensor(train_dat[1][batch_ind].type(torch.LongTensor))
    labels_d = pair_labels(labels)
    save_outputs = []
    Lipschitz_constants = []
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = MetricPreservationLoss(lambda1=criterion_weights[1],lambda2=1).Loss
    for epoch in range(epochs):  # loop over the dataset multiple times
        for i in range(int(len(train_dat[1].T)/batch_sz)):
    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs1, outputs2 = net(inputs)
            loss1 = criterion_weights[0]*criterion1(outputs2, labels)            
            loss2, c = criterion2(outputs1,dmat,labels_d)
            Lipschitz_constants.append(c)
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
    
            # print statistics
            running_loss = loss.item()
            loss1_curve.append(loss1.item())
            loss2_curve.append(loss2.item())
            loss_curve.append(running_loss)
        if epoch%50==0 or epoch==0:
            save_outputs.append(outputs1)
            print(str(running_loss)+': '+str(loss1.item())+' due to CSE loss :'+str(loss2.item())+' due to Topological loss')
    return loss_curve, loss1_curve, loss2_curve, save_outputs, Lipschitz_constants
                
def predict(net,test_dat):
    n_classes = net.n_classes
    n_inputs = len(test_dat)
    n_neurons = net.n_neurons
    X1 = torch.zeros([n_inputs,n_neurons])
    X2 = torch.zeros([n_inputs,n_classes])
    for i in range(n_inputs):
        X1[i], X2[i] = net.forward(test_dat[i])
    return X1.detach().numpy(), X2.detach().numpy()

    

class FunctionalClassifier:
    def __init__(self,kern,n_classes,funct_ = special.hermite):
        self.kern = kern
        self.n_classes = n_classes
        self.funct = funct_
        
    def fit(self,X,labels):
        self.fclass_kernels = [[] for i in range(self.n_classes)]
        highdim_resp = []
        for n in range(len(X)):
            self.fclass_kernels[int(labels[n])].append(self.kern(X[n],np.eye(len(X.T))).pdf)
            highdim_resp.append(self.funct(1+int(labels[n])))
        return self.fclass_kernels, highdim_resp
    
    def map_funcs(self, obj, func_list):
        return [func(obj) for func in func_list]
    
    def evaluate(self,y):
        out = np.zeros([len(y),self.n_classes])
        for i in range(self.n_classes):
            out[:,i] = sum(self.map_funcs(y,self.fclass_kernels[i]))
        return np.argmax(out,1), out
        
class TopologicalBenchmark:
    def __init__(self,nperms,nsamples,npoints,dim,mfld,mfld_noise,phom_params,
                 model,nclasses,criterion,model_params,mfld_args, local=False):
        self.nperms = nperms
        self.nsamples = nsamples
        self.npoints = npoints
        self.dim = dim
        self.mfld = mfld
        self.mfld_noise = mfld_noise
        self.phom_params = phom_params
        self.model = model
        self.nclasses = nclasses
        self.criterion = criterion
        self.model_params = model_params
        self.mfld_args= mfld_args
        self.local = local
        
    def reset_model(self):
        for layer in self.model.children():
           if hasattr(layer, 'reset_parameters'):
               layer.reset_parameters()        
               
    def model_train(self,data):
        self.reset_model()
        dmat = torch.cdist(torch.tensor(data.T),torch.tensor(data.T))
        labels = 0.1*np.ones(len(dmat))
        classes = np.linspace(0,len(dmat),self.nclasses).astype(int)
        for i in range(self.nclasses-1):
            labels[classes[i]:classes[i+1]] = i 
        train_data = [torch.tensor(data),torch.tensor(labels)]
        optimizer = optim.Adam(self.model.parameters())
        loss_curve = train(self.model,train_data,optimizer,[self.model_params[0],self.model_params[1]],
                           dmat,self.model_params[2],self.model_params[3])
        prediction = predict(self.model,train_data[0].T)
        performance = np.sum(labels==np.argmax(prediction[1],1))/len(labels)

        return prediction, labels, performance

    def pers_landscape_diff(self,samples1,samples2):
        pdiags1 = []
        pdiags2 = []
        plands1 = []
        plands2 = []
        Phom = Persistent_Homology()
        for i in range(self.nsamples):
            pdiags1.append(Phom(samples1[i].T,self.phom_params[0],
                                self.phom_params[1],self.phom_params[2],self.phom_params[3])[1])
            pdiags2.append(Phom(samples2[i].T,self.phom_params[0],
                                self.phom_params[1],self.phom_params[2],self.phom_params[3])[1])
            plands1.append(persim.landscapes.PersLandscapeApprox(dgms=pdiags1[-1],hom_deg=self.dim))
            plands2.append(persim.landscapes.PersLandscapeApprox(dgms=pdiags2[-1],hom_deg=self.dim))
        avg_pland1 = persim.landscapes.average_approx(plands1)
        avg_pland2 = persim.landscapes.average_approx(plands2)
        [avg_pland1_snapped, avg_pland2_snapped] = persim.landscapes.snap_pl([avg_pland1, avg_pland2])
        diff_pl = avg_pland1_snapped - avg_pland2_snapped
        significance = diff_pl.sup_norm()
        return plands1, plands2, significance

    def perm_test(self,samples,model_samples):
        pl1, pl2, baseline = self.pers_landscape_diff(samples, model_samples)
        comb_pl = pl1 + pl2
        sup_hist = []
        sig_count = 0
        for i in range(self.nperms):
            A_indices = random.sample(range(2*self.nsamples),self.nsamples)
            B_indices = [_ for _ in range(2*self.nsamples) if _ not in A_indices]
        
            A_pl = [comb_pl[i] for i in A_indices]
            B_pl = [comb_pl[j] for j in B_indices]
        
            A_avg = persim.landscapes.average_approx(A_pl)
            B_avg = persim.landscapes.average_approx(B_pl)
            [A_avg_sn, B_avg_sn] = persim.landscapes.snap_pl([A_avg,B_avg])
        
            shuff_diff = A_avg_sn - B_avg_sn
            sup_diff = shuff_diff.sup_norm()
            sup_hist.append(sup_diff)
            if (shuff_diff.sup_norm() >= baseline): sig_count += 1
        
        pval = sig_count/self.nperms
        return pval, baseline, sup_hist

    def full_test(self):
        gen_samples = [self.mfld(self.npoints,*self.mfld_args)
                       +self.mfld_noise*np.random.randn(*np.shape(self.mfld(self.npoints,*self.mfld_args))) 
                       for i in range(self.nsamples)]
        run_models = [self.model_train(train_mfld) for train_mfld in gen_samples]
        gen_model_samples = [run_models[i][0][0].T for i in range(self.nsamples)]
        labels = [run_models[i][1] for i in range(self.nsamples)]
        performance = [run_models[i][2] for i in range(self.nsamples)]
        
        if self.local:
            perm_tests = [0]*(self.nclasses-1)
            for i in range(self.nclasses-1):
                gen_samp_class = [gen_samples[n][:,labels[n]==i] for n in range(self.nsamples)]
                gen_model_samp_class = [gen_model_samples[n][:,labels[n]==i] for n in range(self.nsamples)]
                perm_tests[i] = self.perm_test(gen_samp_class, gen_model_samp_class)
            return perm_tests, performance
        else:
            return self.perm_test(gen_samples,gen_model_samples), performance
        
class MfldDataset(torch.utils.data.Dataset):
    def __init__(self,dataset):
        self.dataset = dataset
        self.data = dataset.tensors[0]
        self.targets = dataset.tensors[1]
    
        

class HierarchicalDataset(torch.utils.data.Dataset):
    def __init__(self, data,scale_img=True):
        
        if scale_img==True:
            self.x = data.data / 255.0
        else:
            self.x = (data.data-torch.min(data.data))
            self.x = self.x/ torch.max(self.x)
        self.y = data.targets
    
    def __len__(self):
        return len(self.x)
    
    def __hierarchy__(self,partitions):
        self.labels = [[] for i in range(len(partitions))]
        part_mapping = lambda x: np.array([(y in x) for y in self.y])
        for count,partition in enumerate(partitions):
            for idx,part in enumerate(partition):
                self.labels[count].append(idx*part_mapping(part))
            self.labels[count] = sum(self.labels[count])
        return self.labels
    
    def _get_data(self):
        return self.x

    def __getitem__(self, index):
        ys = [self.labels[i][index] for i in range(len(self.labels))]
        img_shape = self.x[index].shape
        return torch.reshape(self.x[index],[img_shape[0]*img_shape[1]]), ys, index
            
class HierarchicalRepresentationNetwork:
    '''Class that wraps around a neural network to output hierarchical representations
    in each sequential layer, it requires hierarchical labeling. The model has to have 
    methods for calling each layer.
    It uses a standard classification loss (CSE/Contrastive loss) and a locally Quasi-Isomorphic
    loss term. See [ref].'''
    def __init__(self,model,savepoints=50,learning_rate=0.001,device='cpu'):
        self.model = model.to(device)
        self.savepoints = savepoints
        self.learning_rate = learning_rate
        self.device = device
        
    def _get_weights_(self):
        weights = []
        for wghts in self.model.layers:
            weights.append(wghts.weight.data)
        return weights
    
    def prop_forward(self,x,stop_layer):
        out = self.model.act_fun(self.model.inp_layer(x.double()))
        for layer in self.model.layers[:-stop_layer]:
            out = self.model.act_fun(layer(out))
        class_out = self.model.class_layer(out)
        return out, class_out
    
    def restricted_backward_pass(self,loss,stop_layer):
        count_layer = 0
        for params in self.model.layers:
            if isinstance(params,nn.Linear):
                params.requires_grad = False
            if count_layer==stop_layer:
                break
            count_layer += 1
        loss.backward()            
        return
        
    def restore_gradients(self):
        for params in self.model.parameters():
            if isinstance(params,nn.Linear):
                params.requires_grad = True
        return
    
    def train(self,data_loader,costs,dmat='standard',epochs=100,plot=True):
        labels = data_loader.dataset.labels
        train_shape = data_loader.dataset.x.shape
        train_data = torch.reshape(data_loader.dataset.x,[train_shape[0],train_shape[1]*train_shape[2]])
        running_loss = [[] for l in range(len(labels))]
        running_loss1 = [[] for l in range(len(labels))]
        running_loss2 = [[] for l in range(len(labels))]
        mflds_train = []
        Lipschitz_constants = []
        if dmat=='standard':
            dmat = torch.cdist(train_data,train_data).to(self.device)
        class_depth = int(self.model.n_layers/len(labels))
        for ep in range(epochs):
            for batch_n, (data, targets,indices) in enumerate(data_loader):
                for branch in range(len(labels)):
                    labels_d = pair_labels(targets[branch])
                    
                    optimizer = optim.Adam(self.model.parameters(),lr=self.learning_rate)
                    optimizer.zero_grad()
                    
                    outs = self.prop_forward(data,class_depth*branch+class_depth)
                    loss1 = costs[branch][0](outs[1],targets[branch])
                    if costs[branch][1]!=0:
                        loss2, c = costs[branch][1](outs[0],dmat[indices,:][:,indices],labels_d)
                        Lipschitz_constants.append(c)
                    else:
                        loss2 = 0
                    loss = (loss1 + loss2)

                    self.restricted_backward_pass(loss,branch*class_depth)
                    optimizer.step()
                    self.restore_gradients()
                    if batch_n%self.savepoints==0 and ep%self.savepoints==0:
                        running_loss[branch].append(loss.item())
                        running_loss1[branch].append(loss1.item()), running_loss2[branch].append(loss2.item())
                        print('Done with '+ str(batch_n)+' batches and '+str(ep)+' epochs on branch ' + str(branch))
                        print(str(loss.item())+': '+str(loss1.item())+' due to CSE loss :'+str(loss2.item())+' due to Isometric loss')
                    
            if ep%self.savepoints==0:
                forward_runs = [self.prop_forward(data, class_depth*i+class_depth) for i in range(len(labels)+1)]
                performance = [torch.sum(torch.argmax(forward_runs[i][1],1)==targets[i])/len(targets[i]) for i in range(len(labels))]
                print('Done with '+str(ep)+' epochs.\==============================')
                print(str(loss.item())+': '+str(loss1.item())+' due to CSE loss :'+str(loss2.item())+' due to Isometric loss')
                print('Classification rate: '+str(performance))
                
        if plot:
            plt.figure()
            for i in range(len(labels)):
                plt.subplot(1, len(labels), i+1)
                plt.plot(running_loss[i])
                plt.plot(running_loss1[i])
                plt.plot(running_loss2[i])
                plt.grid('on'), plt.legend(['Total','CSE','Isometric'])
                plt.title('layer '+str(i))
                plt.xscale('log'), plt.yscale('log')
            plt.tight_layout()
        mflds_train.append(forward_runs)
        return mflds_train, Lipschitz_constants
    
    def test(self,data_loader):
        labels = data_loader.dataset.labels
        performance = [[] for i in range(len(labels))]
        class_depth = int(self.model.n_layers/len(labels))
        for batch_n, (data,targets,indices) in enumerate(data_loader):
            for branch in range(len(labels)):
                out = self.prop_forward(data, class_depth*branch+class_depth)[1]
                performance[branch].append(torch.sum(torch.argmax(out,1)==targets[branch]))
        performance = [sum(performance[i])/data_loader.dataset.__len__() for i in range(len(labels))]
        return performance            

    def fgsm_attack(self, x, targets, epsilon, costs, depth, dmat, labels_d):
        x.requires_grad = True

        out = self.prop_forward(x,depth)
        init_pred = torch.argmax(out[1],1) # get the index of the max out

        # Calculate the loss
        loss1 = costs[0](out[1],targets)
        if costs[1]!=0:
            loss2,_ = costs[1](out[0],dmat,labels_d)
        else:
            loss2 = 0
        loss = (loss1 + loss2)

        self.model.zero_grad()

        loss.backward()
        data_grad = x.grad.data
        sign_data_grad = data_grad.sign()
        perturbed_x = x + epsilon*sign_data_grad
        perturbed_x = torch.clamp(perturbed_x, 0, 1)
        return perturbed_x, init_pred

    def pdg_attack(self, x, targets, epsilon, costs, depth, dmat, labels_d, step_size=0.01, steps=100):
        x.requires_grad = True    
        x_step = x.clone().detach().requires_grad_(True).to(x.device)
        for i in range(steps):
            if i==0:
                x_step_ = x_step.clone().detach().requires_grad_(True)
                prediction = self.prop_forward(x_step_, depth)
                init_pred = torch.argmax(prediction[1],1)
            else: 
                prediction = self.prop_forward(x_step_, depth)
            
            # Calculate the loss
            loss1 = costs[0](prediction[1],targets)
            if costs[1]!=0:
                loss2,_ = costs[1](prediction[0],dmat,labels_d)
            else:
                loss2 = 0
            loss = (loss1 + loss2)
            self.model.zero_grad()
            loss.backward(retain_graph=True)

            with torch.no_grad():
                data_grad = x_step_.grad.sign()              
                x_step += step_size*data_grad
                x_step = torch.max(torch.min(x_step, x + epsilon), x - epsilon)
                x_step = torch.clamp(x_step, 0, 1) 
                if torch.sum((x_step==x+epsilon)+(x_step==x-epsilon))==len(x_step)*len(x_step.T):
                    return x_step, init_pred
        return x_step, init_pred
        
    def hierarchical_test_attack(self,test_loader, costs, epsilon, attack_method='fgsm'):

        labels = test_loader.dataset.labels
        
        # Accuracy counter
        correct = [0 for i in range(len(labels))]
        adv_examples = []
        class_depth = int(self.model.n_layers/len(labels))
        # Loop over all examples in test set
        for batch_n, (data,targets,indices) in enumerate(test_loader):
            for branch in range(len(labels)):
                labels_d = pair_labels(targets[branch])
                dmat = torch.cdist(data,data).double()
                if attack_method=='fgsm':
                    perturbed_data, init_pred = self.fgsm_attack(data, targets[branch],epsilon,
                                                      costs[branch],class_depth*branch+class_depth,
                                                      dmat,labels_d)
                elif attack_method=='pgd':
                    perturbed_data, init_pred = self.pdg_attack(data, targets[branch],epsilon,
                                                      costs[branch],class_depth*branch+class_depth,
                                                      dmat,labels_d)
                else:
                    print('Invalid attack method')
                    return
                
                # Re-classify the perturbed image
                out = self.prop_forward(perturbed_data,class_depth*branch+class_depth)
        
                final_pred = torch.argmax(out[1],1) # get the index of the max output
                correct[branch] += torch.sum(final_pred == targets[branch]).item()
                if batch_n%1000==0:
                    adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                    orig = data.squeeze().detach().cpu().numpy()
                    adv_examples.append((init_pred, final_pred, orig, adv_ex))

        # Calculate final accuracy for this epsilon
        final_acc = [correct[branch]/len(labels[0]) for branch in range(len(labels))]
        return np.vstack(final_acc), adv_examples

def tree_distances(hierarchy):
    tree_depth = len(hierarchy)
    tree_widths = [len(branch) for branch in hierarchy]
    tree_size = sum(tree_widths)
    adjacency_mat = np.zeros([tree_size, tree_size])
    count1 = 0
    for i in range(tree_depth-1):    
        for l in range(tree_widths[i]):
            count2 = sum(tree_widths[:i+1])
            for j in range(tree_widths[i+1]):
                if any(elem in hierarchy[i][l] for elem in hierarchy[i+1][j]):
                    adjacency_mat[count1,count2] = 1
                count2 += 1
            count1+=1
    adjacency_mat = adjacency_mat+adjacency_mat.T
    dmat = shortest_path(adjacency_mat)
    return adjacency_mat, dmat
                                  

class CatDogDataset(Dataset):
    def __init__(self,img_dir,transform):
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.labels[1])
    

    def __hierarchy__(self,depth=2):
        self.file_names = []
        self.labels = [[] for i in range(depth)]
        count = 0
        for root, dirs, files in os.walk(self.img_dir):
            for d in dirs:
                if d=='dogs' or d=='cats':
                    for _,_,files2 in os.walk(os.path.join(root, d)):
                        for f in files2:
                            if not f.startswith('.'):
                                if d=='dogs': self.labels[0].append(0)
                                else: self.labels[0].append(1) 
                else:
                    for _,_,files2 in os.walk(os.path.join(root, d)):
                        for f in files2:
                            if not f.startswith('.'):
                                self.file_names.append(os.path.join(root,d,f))
                                self.labels[1].append(count)
                    count += 1
            
        return self.file_names, self.labels
                
    
    def __getitem__(self, branch_idx):
        img = read_image(self.file_names[branch_idx])  
        try:
            img = self.transform(img).squeeze()
        except:
            img = self.transform[1](img).squeeze()
        label1 = self.labels[0][branch_idx]
        label2 = self.labels[1][branch_idx]
        return img, label1, label2
    
    def get_all(self):
        imgs = []
        for i in range(self.__len__()):
            try:
                imgs.append(self.__getitem__(i)[0])
                if i%1000==0:
                    print(i + ' images loaded')
            except:
                continue
        return imgs, self.labels

def get_distances(dataset,metric=pairwise_distances):
    D = torch.zeros(len(dataset),len(dataset))
    for i in range(len(dataset)):
        for j in range(len(dataset)):
            if i>j:
                D[i,j] = metric(dataset[i],dataset[j])
    D = D+D.T
    return D    
    






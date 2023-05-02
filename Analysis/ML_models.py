# NN classifier
from Algorithms import *
from metrics import *
import os
import torch
from torch import nn
import torch.optim as optim
import numpy as np
import scipy.stats as st
from torch.autograd import Variable
from tqdm import tqdm

# from torchviz import make_dot
from torch.utils.data import Dataset
import torch.nn.functional as FF
from torchvision import datasets
from torchvision.transforms import ToTensor, Resize, Grayscale, Normalize
from torchvision.io import read_image
from scipy import special
from scipy.sparse.csgraph import shortest_path

plt.style.use("default")  # ../misc/report.mplstyle')


class NN_class(nn.Module):
    def __init__(
        self,
        n_neurons,
        n_inputs,
        n_classes,
        projection,
        weights,
        n_layers=1,
        feature_model=None,
    ):
        super(NN_class, self).__init__()
        self.n_neurons = n_neurons
        self.n_inputs = n_inputs
        self.n_classes = n_classes
        self.projection = projection
        self.weights = weights
        self.n_layers = n_layers
        self.feature_model = feature_model
        if self.feature_model != None:
            for param in self.feature_model.parameters():
                param.requires_grad = False

        self.layers = nn.ModuleList()  # []
        self.inp_layer = nn.Linear(n_inputs, n_neurons, bias=True)
        if len(self.projection) > 0:
            self.inp_layer.weight.data = projection
        self.act_fun = nn.Tanh()  #nn.ReLU()# torch.sin##nn.Hardtanh()#
        for i in range(n_layers):
            out_layer = nn.Linear(n_neurons, n_neurons, bias=True)
            # c = torch.tensor(6/self.n_neurons)
            torch.nn.init.orthogonal_(out_layer.weight)#torch.nn.init.uniform_(out_layer.weight,a=-30*torch.sqrt(c),b=30*torch.sqrt(c))#
            if len(self.weights) > 0:
                out_layer.weight.data = weights
            self.layers.append(out_layer.double())
        self.class_layer = nn.Linear(n_neurons, n_classes)

    def forward(self, x):
        if self.feature_model != None:
            x = self.feature_model(x.double())
        out1 = self.act_fun(self.inp_layer(torch.squeeze(x.double())))
        for layer in self.layers:
            out1 = self.act_fun(layer(out1))
        out2 = self.class_layer(out1)
        return out1, out2

    def lim_forward(self, x, layer_stop):
        out1 = self.act_fun(self.inp_layer(x.double()))
        for layer in self.layers[:-layer_stop]:
            out1 = self.act_fun(layer(out1))
        return out1


class Conv_Feature_NN_class(nn.Module):
    def __init__(self, feature_model, n_neurons, n_inputs, n_classes, n_layers=1):
        super(Conv_NN_class, self).__init__()
        self.feature_model = feature_model
        self.n_neurons = n_neurons
        self.n_inputs = n_inputs
        self.n_classes = n_classes
        self.n_layers = n_layers

        self.layers = nn.ModuleList()  # []
        self.inp_layer = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding, bias=True
        )
        conv_size = n_inputs + 2 * padding - kernel_size + 1
        self.pool = nn.MaxPool2d(kernel_size, stride=2)
        pool_size = int(np.floor((conv_size - kernel_size) / 2 + 1))
        self.act_fun = nn.Tanh()  # torch.sin##nn.Hardtanh()#
        self.layers.append(self.pool)
        for i in range(n_layers):
            if i == 0:
                self.layers.append(
                    nn.Linear(
                        pool_size * pool_size * out_channels, n_neurons, bias=True
                    )
                )
            out_layer = nn.Linear(n_neurons, n_neurons, bias=True)
            self.layers.append(out_layer.double())
        self.class_layer = nn.Linear(n_neurons, n_classes)

    def forward(self, x):
        out1 = self.layers[0](self.act_fun(self.inp_layer(x.double())))
        out1 = torch.flatten(out1, 1)
        for layer in self.layers[1:]:
            out1 = self.act_fun(layer(out1))
        out2 = self.class_layer(out1)
        return out1, out2

    def lim_forward(self, x, layer_stop):
        out1 = self.act_fun(self.inp_layer(x.double().T))
        out1 = self.pool(out1)
        for layer in self.layers[:-layer_stop]:
            out1 = self.act_fun(layer(out1))
        return out1


def assign_points(mod_points, mfld_points, asgn_type="rand"):
    if asgn_type == "rand":
        assg = torch.tensor(
            np.random.choice(np.arange(0, len(mod_points)), size=len(mfld_points))
        )
    elif asgn_type == "dist":
        dist = torch.cdist(mod_points, mfld_points.float())
        assg = torch.argmin(dist, 1)
    elif asgn_type == "pairing":
        dist = torch.cdist(mod_points, mfld_points.float())
        sort_dist = torch.min(dist, 1).values
        sort_ids = torch.sort(sort_dist).indices
        assg = torch.zeros(len(dist.T))
        for s in range(len(sort_ids)):
            assg[s] = torch.argmin(dist[:, sort_ids[s]])
            dist[assg[s].long(), :] = torch.inf
    return assg.long()


class MetricPreservationLoss(nn.Module):
    def __init__(self, lambda1, lambda2):
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def Loss(self, y, y_dmats, Y, metric=torch.cdist):
        est_dmats = metric(y, y)
        crit1 = nn.MSELoss()
        # loss = self.lambda1*crit1(Y*est_dmats/(self.lambda2*torch.max(Y*est_dmats)),Y*y1_dmats/torch.max(Y*y1_dmats))**2
        loss = self.lambda1 * crit1(self.lambda2 * Y * est_dmats, Y * y_dmats)
        return loss, (torch.max(Y * y_dmats) / torch.max(Y * est_dmats)).item()


def pair_labels(labels):
    Labels_D = labels[:, None] == labels[None, :]
    # Labels_D[Labels_D==0] = -1
    return Labels_D


def train(net, train_dat, optimizer, criterion_weights, dmat, epochs, batch_sz=20):
    running_loss = 0.0
    loss_curve = []
    loss1_curve = []
    loss2_curve = []
    batch_ind = torch.arange(
        0, len(train_dat[1].T)
    )  # torch.tensor(np.random.choice(np.arange(0,len(train_dat[1].T)),batch_sz,replace=False))
    inputs = torch.tensor(train_dat[0][:, batch_ind])
    inputs.requires_grad = True
    labels = torch.tensor(train_dat[1][batch_ind].type(torch.LongTensor))
    labels_d = pair_labels(labels)
    save_outputs = []
    Lipschitz_constants = []
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = MetricPreservationLoss(lambda1=criterion_weights[1], lambda2=1).Loss    
    pbar = tqdm(range(epochs),ncols=150)

    for epoch in pbar:  # loop over the dataset multiple times
        for batch_n in range(int(len(train_dat[1].T) / batch_sz)):

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs1, outputs2 = net(inputs)
            loss1 = criterion_weights[0] * criterion1(outputs2, labels)
            loss2, c = criterion2(outputs1, dmat, labels_d)
            Lipschitz_constants.append(c)
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss = loss.item()
            loss1_curve.append(loss1.item())
            loss2_curve.append(loss2.item())
            loss_curve.append(running_loss)
            pbar.set_description(f"{batch_n} batches and {epoch} epochs. Loss = {running_loss:.2f}; {loss1.item():.2f} due to CSE loss; {loss2.item(): .2f}: due to Isometric loss.")

        if epoch % 50 == 0 or epoch == 0:
            save_outputs.append(outputs1)

    return loss_curve, loss1_curve, loss2_curve, save_outputs, Lipschitz_constants


def predict(net, test_dat):
    n_classes = net.n_classes
    n_inputs = len(test_dat)
    n_neurons = net.n_neurons
    X1 = torch.zeros([n_inputs, n_neurons])
    X2 = torch.zeros([n_inputs, n_classes])
    for i in range(n_inputs):
        X1[i], X2[i] = net.forward(test_dat[i])
    return X1.detach().numpy(), X2.detach().numpy()


class MfldDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.data = dataset.tensors[0]
        self.targets = dataset.tensors[1]


class HierarchicalDataset(torch.utils.data.Dataset):
    def __init__(self, data, scale_img=True, conv=False):
        self.conv = conv
        if scale_img == True:
            self.x = data.data / 255.0
        else:
            self.x = data.data - torch.min(data.data)
            self.x = self.x / torch.max(self.x)
        if conv:
            self.x = np.swapaxes(self.x, 1, 3)
        if not isinstance(self.x, torch.Tensor):
            self.x = torch.Tensor(self.x)
        self.y = data.targets

    def __len__(self):
        return len(self.x)

    def __hierarchy__(self, partitions):
        self.labels = [[] for i in range(len(partitions))]
        part_mapping = lambda x: np.array([(y in x) for y in self.y])
        for count, partition in enumerate(partitions):
            for idx, part in enumerate(partition):
                self.labels[count].append(idx * part_mapping(part))
            self.labels[count] = sum(self.labels[count])
        return self.labels

    def _get_data(self):
        return self.x

    def __getitem__(self, index):
        ys = [self.labels[i][index] for i in range(len(self.labels))]
        img_shape = self.x[index].shape
        if self.conv:
            return self.x[index], ys, index
        else:
            return torch.reshape(self.x[index], [np.prod(img_shape)]).float(), ys, index


class HierarchicalRepresentationNetwork:
    """Class that wraps around a neural network to output hierarchical representations
    in each sequential layer, it requires hierarchical labeling. The model has to have
    methods for calling each layer.
    It uses a standard classification loss (CSE/Contrastive loss) and a locally Quasi-Isometric
    loss term. 
    See Isometric Representations in Neural Networks Improve Robustness, 
    K Beshkov, J Verhellen & Mikkel Elle Lepperød 2022."""

    def __init__(self, model, savepoints=50, learning_rate=0.001, device="cpu"):
        self.model = model.to(device)
        self.savepoints = savepoints
        self.learning_rate = learning_rate
        self.device = device

    def _get_weights_(self):
        weights = []
        for wghts in self.model.layers:
            weights.append(wghts.weight.data)
        return weights

    def prop_forward(self, x, stop_layer):
        if self.model.n_layers == stop_layer:
            return self.model.forward(x)
        out = self.model.act_fun(self.model.inp_layer(x.double()))
        for layer in self.model.layers[:stop_layer]:
            out = self.model.act_fun(layer(out))
        class_out = self.model.class_layer(out)
        return out, class_out

    def restricted_backward_pass(self, loss, stop_layer):
        count_layer = 0
        for params in self.model.parameters():
            params.requires_grad = False
            if count_layer == stop_layer:
                break
            if len(params.size()) == 2:
                count_layer += 1
        loss.backward()
        return

    def restore_gradients(self):
        for params in self.model.parameters():
            params.requires_grad = True
        return

    def train(self, data_loader, costs, dmat="standard", epochs=100, plot=True):
        labels = data_loader.dataset.labels
        train_shape = data_loader.dataset.x.shape
        train_data = torch.reshape(data_loader.dataset.x,[train_shape[0],np.prod(train_shape[1:])]).to(self.device)
        running_loss = [[] for l in range(len(labels))]
        running_loss1 = [[] for l in range(len(labels))]
        running_loss2 = [[] for l in range(len(labels))]
        mflds_train = []
        Lipschitz_constants = []
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        if dmat == "standard":
            dmat = torch.cdist(train_data, train_data).to(self.device)
        class_depth = int(self.model.n_layers / len(labels))
        performance = [0]*len(labels)
        tot_loss = [0]*len(labels)
        first_loss = [0]*len(labels)
        second_loss = [0]*len(labels)
        pbar = tqdm(range(epochs),ncols=200)
        for ep in pbar:
            for batch_n, (data, targets, indices) in enumerate(data_loader):
                for branch in range(len(labels)):
                    if dmat=='iterate':
                        feat_map = torch.squeeze(self.model.feature_model(data.double().to(self.device)))
                        dmat_temp = torch.cdist(feat_map,feat_map)
                    labels_d = pair_labels(targets[branch]).to(self.device)
                    
                    optimizer.zero_grad()
                    
                    outs = self.prop_forward(data.to(self.device),class_depth*branch+class_depth)
                    loss1 = costs[branch][0](outs[1],targets[branch].to(self.device))
                    if costs[branch][1]!=0:
                        if dmat=='iterate':
                            loss2, c = costs[branch][1](outs[0],dmat_temp,labels_d)
                        else:
                            loss2, c = costs[branch][1](outs[0],dmat[indices,:][:,indices].to(self.device),labels_d)
                        Lipschitz_constants.append(c)
                    else:
                        loss2 = 0
                    loss = loss1 + loss2

                    if len(labels) > 1:
                        self.restricted_backward_pass(loss, branch * class_depth)
                        optimizer.step()
                        self.restore_gradients()
                    else:
                        loss.backward()
                        optimizer.step()

                    if batch_n % self.savepoints == 0 or ep % self.savepoints == 0:
                        running_loss[branch].append(round(loss.item(),2))
                        running_loss1[branch].append(round(loss1.item(),2)), running_loss2[
                            branch
                        ].append(round(loss2.item(),2))
                        
                    if batch_n%self.savepoints==0 or ep%self.savepoints==0:
                        running_loss[branch].append(loss.item())
                        running_loss1[branch].append(loss1.item()), running_loss2[branch].append(loss2.item())
                        print('Done with '+ str(batch_n)+' batches and '+str(ep)+' epochs on branch ' + str(branch))
                        print(str(loss.item())+': '+str(loss1.item())+' due to CSE loss :'+str(loss2.item())+' due to Isometric loss')
                    
            if ep%self.savepoints==0:
                forward_runs = [self.prop_forward(data.to(self.device), class_depth*branch+class_depth) for i in range(1,len(labels)+1)]
                performance = [torch.sum(torch.argmax(forward_runs[i][1],1)==targets[i].to(self.device))/len(targets[i].to(self.device)) for i in range(len(labels))]
                print('Done with '+str(ep)+' epochs.\==============================')
                print(str(loss.item())+': '+str(loss1.item())+' due to CSE loss :'+str(loss2.item())+' due to Isometric loss')
                print('Classification rate: '+str(performance))
                
        if plot:
            plt.figure()
            for i in range(len(labels)):
                plt.subplot(1, len(labels), i + 1)
                plt.plot(running_loss[i])
                plt.plot(running_loss1[i])
                plt.plot(running_loss2[i])
                plt.grid("on"), plt.legend(["Total", "CSE", "Isometric"])
                plt.title("layer " + str(i))
                plt.xscale("log"), plt.yscale("log")
            plt.tight_layout()
        mflds_train.append(forward_runs)
        return mflds_train, Lipschitz_constants

    def test(self, data_loader):
        labels = data_loader.dataset.labels
        performance = [[] for i in range(len(labels))]
        class_depth = int(self.model.n_layers / len(labels))
        for batch_n, (data, targets, indices) in enumerate(data_loader):
            for branch in range(len(labels)):
                out = self.prop_forward(data.to(self.device), class_depth*branch+class_depth)[1]
                performance[branch].append(torch.sum(torch.argmax(out,1)==targets[branch].to((self.device))))
        performance = [sum(performance[i])/data_loader.dataset.__len__() for i in range(len(labels))]
        return performance            

    def fgsm_attack(self, x, targets, epsilon, costs, depth, dmat, labels_d):
        x.requires_grad = True

        out = self.prop_forward(x, depth)
        init_pred = torch.argmax(out[1], 1)  # get the index of the max out

        # Calculate the loss
        loss1 = costs[0](out[1], targets)
        if costs[1] != 0:
            loss2, _ = costs[1](out[0], dmat, labels_d)
        else:
            loss2 = 0
        loss = loss1 + loss2

        self.model.zero_grad()

        loss.backward()
        data_grad = x.grad.data
        sign_data_grad = data_grad.sign()
        perturbed_x = x + epsilon * sign_data_grad
        perturbed_x = torch.clamp(perturbed_x, 0, 1)
        return perturbed_x, init_pred

    def pdg_attack(
        self, x, targets, epsilon, costs, depth, dmat, labels_d, ball_size=0.5
    ):
        steps = 10
        x.requires_grad = True
        x_step = x.clone().detach().requires_grad_(True).to(x.device)
        for i in range(steps):
            if i == 0:
                x_step_ = x_step.clone().detach().requires_grad_(True)
                prediction = self.prop_forward(x_step_, depth)
                init_pred = torch.argmax(prediction[1], 1)
            else:
                prediction = self.prop_forward(x_step_, depth)

            # Calculate the loss
            loss1 = costs[0](prediction[1], targets)
            if costs[1] != 0:
                loss2, _ = costs[1](prediction[0], dmat, labels_d)
            else:
                loss2 = 0
            loss = loss1 + loss2
            self.model.zero_grad()
            loss.backward(retain_graph=True)

            with torch.no_grad():
                data_grad = x_step_.grad.sign()
                x_step += epsilon * data_grad
                x_step = torch.max(torch.min(x_step, x + ball_size), x - ball_size)
                x_step = torch.clamp(x_step, 0, 1)
                if torch.sum((x_step == x + epsilon) + (x_step == x - epsilon)) == len(
                    x_step
                ) * len(x_step.T):
                    return x_step, init_pred
        return x_step, init_pred
        
    def hierarchical_test_attack(self,test_loader, costs, epsilon, attack_method='fgsm',featurize=False):

        labels = test_loader.dataset.labels
        labels_d = [pair_labels(labels[branch]) for branch in range(len(labels))]
        # Accuracy counter
        correct = [0 for i in range(len(labels))]
        adv_examples = []
        class_depth = int(self.model.n_layers / len(labels))
        # Loop over all examples in test set
        for batch_n, (data, targets, indices) in enumerate(test_loader):
            for branch in range(len(labels)):
                labels_d = pair_labels(targets[branch])
                if featurize:
                    feat_map =  torch.squeeze(self.model.feature_model(data.double().to(self.device)))
                    dmat = torch.cdist(feat_map,feat_map).double()
                else:
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
                out = self.prop_forward(
                    perturbed_data, class_depth * branch + class_depth
                )

                final_pred = torch.argmax(out[1], 1)  # get the index of the max output
                correct[branch] += torch.sum(final_pred == targets[branch]).item()
                if batch_n % 1000 == 0:
                    adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                    orig = data.squeeze().detach().cpu().numpy()
                    adv_examples.append((init_pred, final_pred, orig, adv_ex))

        # Calculate final accuracy for this epsilon
        final_acc = [correct[branch] / len(labels[0]) for branch in range(len(labels))]
        return np.vstack(final_acc), adv_examples


def tree_distances(hierarchy):
    tree_depth = len(hierarchy)
    tree_widths = [len(branch) for branch in hierarchy]
    tree_size = sum(tree_widths)
    adjacency_mat = np.zeros([tree_size, tree_size])
    count1 = 0
    for i in range(tree_depth - 1):
        for l in range(tree_widths[i]):
            count2 = sum(tree_widths[: i + 1])
            for j in range(tree_widths[i + 1]):
                if any(elem in hierarchy[i][l] for elem in hierarchy[i + 1][j]):
                    adjacency_mat[count1, count2] = 1
                count2 += 1
            count1 += 1
    adjacency_mat = adjacency_mat + adjacency_mat.T
    dmat = shortest_path(adjacency_mat)
    return adjacency_mat, dmat


def compute_featurized_distances(model, datapoints, targets):
    dmat = torch.zeros([len(datapoints), len(datapoints)])
    classes = torch.unique(torch.Tensor(targets))
    for c in classes:
        class_inds = torch.where(torch.Tensor(targets) == c)[0]
        class_out = torch.squeeze(model(datapoints[class_inds]))
        d_class = torch.cdist(class_out, class_out)
        for i in range(len(d_class)):
            for j in range(len(d_class)):
                if i > j:
                    dmat[class_inds[i], class_inds[j]] = d_class[i, j]
        print(c)
    return dmat.T + dmat

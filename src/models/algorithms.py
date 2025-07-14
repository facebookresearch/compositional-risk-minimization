# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
from torch.nn import functional as F
from torch.nn.functional import cross_entropy, softmax, one_hot
import torch.autograd as autograd
import numpy as np
import pdb

def get_algorithm(hparams, net, optim):

    # 'RWG' and 'SUBG' are both ERM but they differ in how they balance batches
    if hparams['algorithm_name'] in ['ERM', 'RWG', 'SUBG']:
        return ERM(hparams, net, optim)
    elif hparams['algorithm_name'] == 'GroupDRO':
        return GroupDRO(hparams, net, optim)
    elif hparams['algorithm_name'] == 'IRM':
        return IRM(hparams, net, optim)
    elif hparams['algorithm_name'] in ['LA_Group', 'LA_Cond']:
            return LA(hparams, net, optim)    
    elif hparams['algorithm_name'] in ['CRM']:
        return CRM(hparams, net, optim)
    elif hparams['algorithm_name'] in ['CRM_Multi_Attr']:
        return CRMMultiAttr(hparams, net, optim)
    elif hparams['algorithm_name'] == 'VREx':
        return VREx(hparams, net, optim)
    elif hparams['algorithm_name'] == 'Fishr':
        return Fishr(hparams, net, optim)
    elif hparams['algorithm_name'] == 'Mixup':
        return Mixup(hparams, net, optim)
    elif hparams['algorithm_name'] == 'Group_ERM':
        return Group_ERM(hparams, net, optim)

class CRM:
    def __init__(self, hparams, net, optim):
        self.device = hparams["device"]
        self.hparams = hparams
        self.net = net.to(self.device)
        self.optim = optim

        self.num_classes= hparams['num_y']
        self.num_attr= hparams['num_m']

    def get_loss(self, y_hat, y, m=None):
        return cross_entropy(y_hat, y.view(-1).long())
        
    def get_crm_offset(self):
        if self.hparams['precompute_features']:
            return self.net.offset
        elif self.hparams['net_type'] == 'resnet':
            return self.net.fc.offset            
        elif self.hparams['net_type'] == 'bert':
            return self.net.net.classifier.offset        
        elif self.hparams['net_type'] == 'mlp':
            return self.net.fc.offset
        elif self.hparams['net_type'] == 'linear':
            return self.net.fc.offset

    def get_crm_extrapolate_bias(self, loader, discarded_groups= []):
        "Compute the extrapolated bias for CRM based on equation (11) in the paper."
        with torch.no_grad():
            energy_tr=[]
            for batch in loader:
                _, x, _, _= batch
                x= x.to(self.device)
                energy_tr.append( self.net.forward(x) )
            energy_tr= torch.concatenate(energy_tr, dim=0)        
            offset= self.get_crm_offset()
            logits_tr= self.get_crm_logits(energy_tr, offset, self.train_log_prior_prob)
            logits_tr= torch.exp(logits_tr).view(logits_tr.shape[0], self.num_classes * self.num_attr)
            for g_idx in discarded_groups:
                logits_tr[:, g_idx]= 0
            logits_tr= torch.sum(logits_tr, dim=1).unsqueeze(-1).unsqueeze(-1)
            self.offset_corr= torch.log( torch.mean( torch.exp(-energy_tr) / logits_tr , dim=0) )
        return 

    def get_crm_logits(self, energy_mat: torch.Tensor, offset: torch.Tensor, log_prior_prob: torch.Tensor):
        """ Compute logits as per the additive energy classifier
        Input:
            energy_mat: Tensor of shape (batch_size, num_classes, num_attr)
            offset: Tensor of shape (num_classes, num_attr)
            log_prior_prob: Tensor of shape (num_classes * num_attr) representing log prior probabilities            
        Output:
            logits: Tensor of shape (batch_size, num_classes, num_attr)
        """
        return -energy_mat - offset + log_prior_prob.view(self.num_classes, self.num_attr)

    def update(self, batch):
        _, x, y, m = batch
        x = x.to(self.device)
        y = y.to(self.device)
        m = m.to(self.device)
        self.optim.zero_grad(set_to_none=True)

        #Forward Pass
        logit= self.net.forward(x)
        offset= self.get_crm_offset()
        logit= self.get_crm_logits(logit, offset, self.train_log_prior_prob)

        #Compute group labels and then compute cross entropy loss in predicting the group labels
        g= self.num_attr * y + m
        #Reshape (batch size, num classes, num attributes) to (batch size, num classes * num attributes)
        logit= logit.view(logit.shape[0], self.num_classes * self.num_attr )
        loss = F.cross_entropy(logit, g.long(), reduction='mean')

        #Backproagation
        loss.backward()
        self.optim.step()
        if self.optim.lr_scheduler is not None:
            self.optim.lr_scheduler.step()

        return loss.item()

    def evaluate(self, loader, log_prior_prob: torch.Tensor,  extrapolate: bool= False):
        """
        Evaluate the model on the given loader.
        Input:
            loader: DataLoader object containing the data to evaluate on.
            log_prior_prob: Tensor of shape (num_classes * num_attr) representing log prior probabilities at test time.
            extrapolate: Boolean indicating whether to use extrapolated biases (self.get_crm_extrapolate_bias()) or learned biases (self.get_crm_offset()).
        Output:
            ys: Tensor of true labels.
            y_hats: Tensor of predicted labels.
            ms: Tensor of spurious attributes.            
        """

        self.net.eval()
        i_s = []
        ys = []
        y_hats = []
        ms = []
        with torch.no_grad():
            if extrapolate:
                offset= self.offset_corr
            else:
                offset= self.get_crm_offset()

            for batch in loader:
                i, x, y, m = batch
                x = x.to(self.device)
                i_s += [i]
                ys += [y.to(self.device)]

                #Forward Pass
                logit= self.net.forward(x)
                logit= self.get_crm_logits(logit, offset, log_prior_prob)
                
                #Get label predictions by marginalizing over the attributes
                logit= logit.view(logit.shape[0], self.num_classes * self.num_attr)
                y_pred= torch.softmax(logit, dim=1)        
                y_pred= y_pred.view(y_pred.shape[0], self.num_classes, self.num_attr )
                y_pred = torch.sum(y_pred, dim=2)                

                y_hats+= [y_pred]
                ms += [m.to(self.device)]                

        i_s = torch.cat(i_s)
        sorted_indices = torch.argsort(i_s)
        ys = torch.cat(ys)[sorted_indices].view(-1)
        y_hats = torch.cat(y_hats)[sorted_indices]
        ms = torch.cat(ms)[sorted_indices].view(-1)
        self.net.train()
        return ys, y_hats, ms
    

class CRMMultiAttr():

    def __init__(self, hparams, net, optim):
        self.device = hparams["device"]
        self.hparams = hparams
        self.net = net.to(self.device)
        self.optim = optim

        self.num_classes= hparams['num_y']
        self.num_attr= hparams['num_m']
        self.total_spur_attr= int(np.log2(self.num_attr))
        self.feat_shape= (self.num_classes,) + self.total_spur_attr * (2,)

    def get_group_label(self, y, m):
        #Assumes all spurious attributes are binary
        return (2**self.total_spur_attr) * y + m

    def get_loss(self, y_hat, y, m=None):
        return cross_entropy(y_hat, y.view(-1).long())

    def get_crm_offset(self):
        if self.hparams['precompute_features']:
            return self.net.offset
        elif self.hparams['net_type'] == 'resnet':
            return self.net.fc.offset            
        elif self.hparams['net_type'] == 'bert':
            return self.net.net.classifier.offset        
        elif self.hparams['net_type'] == 'mlp':
            return self.net.fc.offset
        
    def get_crm_extrapolate_bias(self, loader, discarded_groups= []):
        "Compute the extrapolated bias for CRM based on equation (11) in the paper."
        with torch.no_grad():
            energy_tr=[]
            for batch in loader:
                _, x, _, _= batch
                x= x.to(self.device)
                energy_tr.append( self.net.forward(x) )
            energy_tr= torch.concatenate(energy_tr, dim=0)        

            offset= self.get_crm_offset()
            logits_tr= self.get_crm_logits(energy_tr, offset, self.train_log_prior_prob)
            #Collapse along multi attr dimension to get group probs
            logits_tr= torch.exp(logits_tr).view(logits_tr.shape[0], -1)
            for g_idx in discarded_groups:
                logits_tr[:, g_idx]= 0
            logits_tr= torch.sum(logits_tr, dim=1)
            #Unsqueeze along multi attr dimensions to allow for same total dimensions as energy_tr
            logits_tr= logits_tr.view( (logits_tr.shape[0],) + (1+self.total_spur_attr) * (1,) )
            assert len(energy_tr.shape) == len(logits_tr.shape)

            self.offset_corr= torch.log( torch.mean( torch.exp(-energy_tr) / logits_tr , dim=0) )
        
        return 

    def get_crm_logits(self, energy_mat: torch.Tensor, offset: torch.Tensor, log_prior_prob: torch.Tensor):
        """ Compute logits as per the additive energy classifier
        Input:
            energy_mat: Tensor of shape (batch_size, num_classes, num_attr)
            offset: Tensor of shape (num_classes, num_attr)
            log_prior_prob: Tensor of shape (num_classes * num_attr) representing log prior probabilities            
        Output:
            logits: Tensor of shape (batch_size, num_classes, num_attr)
        """        
        return -energy_mat - offset + log_prior_prob.view(self.feat_shape)

    def update(self, batch):
        _, x, y, m = batch
        x = x.to(self.device)
        y = y.to(self.device)
        m = m.to(self.device)
        self.optim.zero_grad(set_to_none=True)

        #Forward Pass
        logit= self.net.forward(x)
        assert logit.shape[1:] == self.feat_shape
        offset= self.get_crm_offset()
        logit= self.get_crm_logits(logit, offset, self.train_log_prior_prob)

        #Compute group labels and then compute cross entropy loss in predicting the group labels
        g= self.get_group_label(y,m)

        #Collapse along multi attr dimensions to get group probs
        logit= logit.view(logit.shape[0], -1)
        loss = F.cross_entropy(logit, g.long(), reduction='mean')        

        #Backproagation
        loss.backward()
        self.optim.step()
        if self.optim.lr_scheduler is not None:
            self.optim.lr_scheduler.step()
    

    def evaluate(self, loader, log_prior_prob: torch.Tensor,  extrapolate: bool= False):
        """
        Evaluate the model on the given loader.
        Input:
            loader: DataLoader object containing the data to evaluate on.
            log_prior_prob: Tensor of shape (num_classes * num_attr) representing log prior probabilities at test time.
            extrapolate: Boolean indicating whether to use extrapolated biases (self.get_crm_extrapolate_bias()) or learned biases (self.get_crm_offset()).
        Output:
            ys: Tensor of true labels.
            y_hats: Tensor of predicted labels.
            ms: Tensor of spurious attributes.            
        """
        
        self.net.eval()
        i_s = []
        ys = []
        y_hats = []
        ms = []
        with torch.no_grad():
            if extrapolate:
                offset= self.offset_corr
            else:
                offset= self.get_crm_offset()

            for batch in loader:
                i, x, y, m = batch
                x = x.to(self.device)
                i_s += [i]
                ys += [y.to(self.device)]

                #Forward Pass
                logit= self.net.forward(x)
                logit= self.get_crm_logits(logit, offset, log_prior_prob)
                
                #Get Label Predictions
                #Collapse along multi attr dimensions to get group probs
                logit= logit.view(logit.shape[0], -1)
                y_pred= torch.softmax(logit, dim=1)

                #Expand along multi attr dimension
                y_pred= y_pred.view( (y_pred.shape[0],) + self.feat_shape )

                #Sum along spurious attr dimensions
                dims_to_sum = tuple(dim_idx for dim_idx in range(y_pred.ndim) if dim_idx not in [0, 1] )
                assert len(dims_to_sum) == self.total_spur_attr
                y_pred = torch.sum(y_pred, dim=dims_to_sum)                

                y_hats+= [y_pred]
                ms += [m.to(self.device)]                

        i_s = torch.cat(i_s)
        sorted_indices = torch.argsort(i_s)
        ys = torch.cat(ys)[sorted_indices].view(-1)
        y_hats = torch.cat(y_hats)[sorted_indices]
        ms = torch.cat(ms)[sorted_indices].view(-1)
        self.net.train()
        return ys, y_hats, ms

class ERM:
    def __init__(self, hparams, net, optim):
        self.device = hparams["device"]
        self.hparams = hparams
        self.net = net.to(self.device)
        self.optim = optim

    def get_loss(self, y_hat, y, m=None):
        return cross_entropy(y_hat, y.view(-1).long())

    def update(self, batch):
        _, x, y, m = batch
        x = x.to(self.device)
        y = y.to(self.device)
        m = m.to(self.device)
        self.optim.zero_grad(set_to_none=True)
        loss = self.get_loss(self.net(x), y, m)
        loss.backward()
        self.optim.step()
        if self.optim.lr_scheduler is not None:
            self.optim.lr_scheduler.step()
        
        return loss.item()

    def predict(self, x):
        return self.net(x)            

    def evaluate(self, loader):
        self.net.eval()
        i_s = []
        ys = []
        y_hats = []
        ms = []
        with torch.no_grad():
            for batch in loader:
                i, x, y, m = batch
                x = x.to(self.device)
                i_s += [i]
                ys += [y.to(self.device)]
                y_hats += [self.predict(x)]
                ms += [m.to(self.device)]
        i_s = torch.cat(i_s)
        sorted_indices = torch.argsort(i_s)
        ys = torch.cat(ys)[sorted_indices].view(-1)
        y_hats = torch.cat(y_hats)[sorted_indices]
        ms = torch.cat(ms)[sorted_indices].view(-1)
        self.net.train()
        return ys, y_hats, ms


class GroupDRO(ERM):
    def __init__(self, hparams, net, optim):
        super(GroupDRO, self).__init__(hparams, net, optim)
        self.eta = hparams['eta']
        self.q = torch.ones(hparams['num_y'] * hparams['num_m'])

    def get_loss(self, y_hat, y, m=None):
        grp_losses = torch.zeros(len(self.q))
        g = (self.hparams['num_m'] * y + m).view(-1)
        losses = cross_entropy(y_hat, y.view(-1).long(), reduction='none')
        for i in g.unique().int():
            grp_losses[i] = losses[g == i].mean()
            self.q[i] *= (self.eta * grp_losses[i]).exp().item()

        self.q /= self.q.sum()
        return (self.q * grp_losses).sum()


class Mixup(ERM):
    
    def __init__(self, hparams, net, optim):
        super(Mixup, self).__init__(hparams, net, optim)
        self.alpha= hparams['alpha']

        #Placeholder for extracted features
        self.extracted_features= None

        #Register the forward hook
        if not self.hparams['precompute_features']:
            if self.hparams['net_type'] == 'resnet':
                self.hook= self.net.fc.register_forward_hook(self.get_features)
            elif self.hparams['net_type'] == 'bert':
                self.hook= self.net.net.classifier.register_forward_hook(self.get_features)
            else:
                raise NotImplementedError

    def get_features(self, module, input, output):
        "Hook to extract features from the penultimate layer."
        self.extracted_features = input[0]

    def mixup_data(self, x, y):
        lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 1

        batch_size = x.shape[0]
        index = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]

        return mixed_x, y_a, y_b, lam

    def get_loss(self, x, y, m=None):
        #Get Features
        if self.hparams['precompute_features']:
            feat= x        
        else:
            _= self.net(x)
            feat= self.extracted_features

        #Mixup
        feat_mixup, yi, yj, lam= self.mixup_data(feat, y)

        #Classifier Predictions    
        if self.hparams['precompute_features']:
            y_pred= self.net(feat_mixup)
        elif self.hparams['net_type'] == 'resnet':
            y_pred= self.net.fc(feat_mixup)
        elif self.hparams['net_type'] == 'bert':
            y_pred= self.net.net.classifier(feat_mixup)
        
        #Loss
        loss_mixup = lam * F.cross_entropy(y_pred, yi) + (1 - lam) * F.cross_entropy(y_pred, yj)
        return loss_mixup

    def update(self, batch):
        _, x, y, m = batch
        x = x.to(self.device)
        y = y.to(self.device)
        m = m.to(self.device)
        self.optim.zero_grad(set_to_none=True)
                
        loss = self.get_loss(x, y, m)
        loss.backward()
        self.optim.step()
        if self.optim.lr_scheduler is not None:
            self.optim.lr_scheduler.step()


class IRM(ERM):
    def __init__(self, hparams, net, optim):
        super(IRM, self).__init__(hparams, net, optim)
        self.update_count = torch.tensor([0])

    @staticmethod
    def _irm_penalty(logits, y):
        device = "cuda" if logits[0][0].is_cuda else "cpu"
        scale = torch.tensor(1.).to(device).requires_grad_()
        loss_1 = cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def get_loss(self, y_hat, y, m=None):
        penalty_weight = (self.hparams['lambda'] if self.update_count
                          >= self.hparams['penalty_anneal_iters'] else 1.0)
        nll = 0.
        penalty = 0.

        for i in m.unique().int():
            nll += cross_entropy(y_hat[m == i], y[m == i])
            penalty += self._irm_penalty(y_hat[m == i], y[m == i])
        nll /= len(m.unique())
        penalty /= len(m.unique())
        loss = nll + (penalty_weight * penalty)

        self.update_count += 1
        return loss

class Group_ERM(ERM):

    def get_loss(self, out, y, m=None):        
        y_hat = out[..., :self.hparams['num_y']]  
        m_hat = out[..., self.hparams['num_y']:]
        return 0.5* ( cross_entropy(y_hat, y) + cross_entropy(m_hat, m) )

    def evaluate(self, loader):
        self.net.eval()
        i_s = []
        ys = []
        y_hats = []
        ms = []
        with torch.no_grad():
            for batch in loader:
                i, x, y, m = batch
                x = x.to(self.device)
                i_s += [i]
                ys += [y.to(self.device)]
                y_hats += [self.net(x)[..., :self.hparams['num_y']]]
                ms += [m.to(self.device)]
        i_s = torch.cat(i_s)
        sorted_indices = torch.argsort(i_s)
        ys = torch.cat(ys)[sorted_indices].view(-1)
        y_hats = torch.cat(y_hats)[sorted_indices]
        ms = torch.cat(ms)[sorted_indices].view(-1)
        self.net.train()

        return ys, y_hats, ms

class LA(ERM):

    def predict(self, x):
        out = self.net(x)
        if self.net.training:
            return out
        return out[..., :self.hparams['num_y']]

    def gce(self, logits, target, tau=0.8):
        probs = softmax(logits, dim=1)
        pred = probs.argmax(1)
        target_one_hot = one_hot(target, num_classes=logits.size(1)).float()
        loss = (1 - (probs ** tau) * target_one_hot).sum(dim=1) / tau

        return sum([loss[pred.eq(t)].mean() for t in target.unique()])

    def get_m_y_prior(self, y, m_prob=None):
        # LC: \sum p(m,y|x) = p(m,y)
        prior = torch.cat([
            m_prob[y.eq(y_i)].sum(0).unsqueeze(1) / len(y)
            for y_i in range(self.hparams['num_y'])], 1)        # Christos Tsirigotis et al.: p(y|m)
        if self.hparams['adjustment_method'] == 'uLA':
            return prior / prior.sum(1, keepdim=True)
        elif self.hparams['adjustment_method'] == 'LC':
            return prior
        else:
            raise NotImplementedError

    def get_loss(self, out, y, m=None):
        y_hat = out[..., :self.hparams['num_y']]
        loss = 0
        if self.hparams['use_true_m']:
            m_prob = one_hot(m).float()
        else:
            m_hat = out[..., self.hparams['num_y']:]
            m_prob = softmax(m_hat.detach() / self.hparams['temp'], dim=1)
            loss += 0.1 * self.gce(m_hat, y)

        m_y_prior = self.get_m_y_prior(y, m_prob)
        m_pred = m_prob.argmax(1)
        p_y_given_m = m_y_prior[m_pred]
        shift = torch.log(p_y_given_m + 1e-4)

        loss += cross_entropy(y_hat + shift, y)

        return loss


class VREx(ERM):
    def __init__(self, hparams, net, optim):
        super(VREx, self).__init__(hparams, net, optim)
        self.update_count = torch.tensor([0])

    def get_loss(self, y_hat, y, m=None):
        penalty_weight = (self.hparams['lambda'] if self.update_count
                          >= self.hparams['penalty_anneal_iters'] else 1.0)
        grp_losses = []
        g = (self.hparams['num_m'] * y + m).view(-1)
        losses = cross_entropy(y_hat, y.view(-1).long(), reduction='none')

        for i in g.unique().int():
            grp_losses.append(losses[g == i].mean())

        avg_loss = sum(grp_losses) / len(grp_losses)
        if len(grp_losses) == 1:
            #Just append average loss to avoid issues while with backpropagation on the torch.var() on single element array
            grp_losses.append(avg_loss)
        variance_penalty = torch.var(torch.stack(grp_losses))
        loss = avg_loss + penalty_weight * variance_penalty
        return loss


class Fishr(ERM):
    def __init__(self, hparams, net, optim):
        super(Fishr, self).__init__(hparams, net, optim)
        self.update_count = torch.tensor([0])

    def get_env_gradients(self, loss, params):
        grads = autograd.grad(loss, params, retain_graph=True, create_graph=True)
        return torch.cat([grad.view(-1) for grad in grads])

    def get_loss(self, y_hat, y, m=None):
        penalty_weight = (self.hparams['lambda'] if self.update_count
                          >= self.hparams['penalty_anneal_iters'] else 1.0)
        grp_losses = []
        g = (self.hparams['num_m'] * y + m).view(-1)
        losses = cross_entropy(y_hat, y.view(-1).long(), reduction='none')

        env_gradients = []
        params = list(self.net.parameters())

        for i in g.unique().int():
            grp_loss = losses[g == i].mean()
            grp_losses.append(grp_loss)
            env_gradients.append(self.get_env_gradients(grp_loss, params))

        avg_loss = sum(grp_losses) / len(grp_losses)
        avg_gradient = torch.mean(torch.stack(env_gradients), dim=0)
        penalty = sum([(grad - avg_gradient).pow(2).sum() for grad in env_gradients]) / len(env_gradients)

        return avg_loss + penalty_weight * penalty

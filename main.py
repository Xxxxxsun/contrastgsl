import argparse
import copy
from datetime import datetime

import time
import numpy as np
from copy import deepcopy
import torch
import torch.nn.functional as F

from data_loader import load_data
#from data_loader import load_ptbdata
from model import GCN, GCL
from graph_learners import *
from utils import *
from sklearn.cluster import KMeans
import dgl

from deeprobust.graph.defense.pgd import PGD, prox_operators

import random

EOS = 1e-10

class EstimateAdj(nn.Module):
    """Provide a pytorch parameter matrix for estimated
    adjacency matrix and corresponding operations.
    """

    def __init__(self, adj, symmetric=False, device='cpu'):
        super(EstimateAdj, self).__init__()
        n = len(adj)
        self.estimated_adj = nn.Parameter(torch.FloatTensor(n, n))
        self._init_estimation(adj)
        self.symmetric = symmetric
        self.device = device

    def _init_estimation(self, adj):
        with torch.no_grad():
            n = len(adj)
            self.estimated_adj.data.copy_(adj)

    def forward(self):
        return self.estimated_adj

    def normalize(self):

        if self.symmetric:
            adj = (self.estimated_adj + self.estimated_adj.t())/2
        else:
            adj = self.estimated_adj

        normalized_adj = self._normalize(adj + torch.eye(adj.shape[0]).to(self.device))
        return normalized_adj

    def _normalize(self, mx):
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1/2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
        mx = mx @ r_mat_inv
        return mx




class Experiment:
    def __init__(self):
        super(Experiment, self).__init__()


    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        np.random.seed(seed)
        random.seed(seed)
        dgl.seed(seed)
        dgl.random.seed(seed)


    def loss_cls(self, model, mask, features, labels):
        logits = model(features)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[mask], labels[mask], reduction='mean')
        accu = accuracy(logp[mask], labels[mask])
        return loss, accu


    def loss_gcl(self, model, graph_learner, features, anchor_adj,adjj,t):

        # view 1: anchor graph
        if args.maskfeat_rate_anchor:
            mask_v1, _ = get_feat_mask(features, args.maskfeat_rate_anchor)
            features_v1 = features * (1 - mask_v1)
        else:
            features_v1 = copy.deepcopy(features)

        z1, _ = model(features_v1, anchor_adj, 'anchor')

        # view 2: learned graph
        if args.maskfeat_rate_learner:
            mask, _ = get_feat_mask(features, args.maskfeat_rate_learner)
            features_v2 = features * (1 - mask)
        else:
            features_v2 = copy.deepcopy(features)

        learned_adj = graph_learner(features)
        learned_adj = learned_adj * t + (1-t) * adjj
        #print("adjtype:",type(learned_adj))
        if not args.sparse:
            learned_adj = symmetrize(learned_adj)
            learned_adj = normalize(learned_adj, 'sym', args.sparse)

        z2, _ = model(features_v2, learned_adj, 'learner')

        # compute loss
        if args.contrast_batch_size:
            node_idxs = list(range(features.shape[0]))
            # random.shuffle(node_idxs)
            batches = split_batch(node_idxs, args.contrast_batch_size)
            loss = 0
            for batch in batches:
                weight = len(batch) / features.shape[0]
                loss += model.calc_loss(z1[batch], z2[batch]) * weight
        else:
            loss = model.calc_loss(z1, z2)

        return loss, learned_adj
    
    
    def train_adj_bypgnn(self, epoch, features, adj,args):
        estimator = self.estimator
        #args = self.args
        if args.debug:
            print("\n=== train_adj ===")
        t = time.time()
        estimator.train()
        self.optimizer_adj.zero_grad()

        loss_l1 = torch.norm(estimator.estimated_adj, 1)
        if loss_l1 > 11000:
            loss_fro = torch.norm(estimator.estimated_adj - adj, p='fro')
            normalized_adj = estimator.normalize()

            if args.lambda_:
                loss_smooth_feat = self.feature_smoothing(estimator.estimated_adj, features)
            else:
                loss_smooth_feat = 0 * loss_l1
            #delete gcnloss
            '''
            output = self.model(features, normalized_adj)
            loss_gcn = F.nll_loss(output[idx_train], labels[idx_train])
            acc_train = accuracy(output[idx_train], labels[idx_train])
            '''
            loss_symmetric = torch.norm(estimator.estimated_adj \
                            - estimator.estimated_adj.t(), p="fro")

            loss_diffiential =  loss_fro + args.lambda_ * loss_smooth_feat + args.phi * loss_symmetric   #args.gamma * loss_gcn + 
        
        

            loss_diffiential.backward()

            self.optimizer_adj.step()
            loss_nuclear =  0 * loss_fro
            if args.beta != 0:
                self.optimizer_nuclear.zero_grad()
                self.optimizer_nuclear.step()
                loss_nuclear = prox_operators.nuclear_norm

            self.optimizer_l1.zero_grad()
            self.optimizer_l1.step()

            total_loss = loss_fro \
                        + args.alpha * loss_l1 \
                        + args.beta * loss_nuclear \
                        + args.phi * loss_symmetric
                        #+ args.gamma * loss_gcn \

            estimator.estimated_adj.data.copy_(torch.clamp(
                    estimator.estimated_adj.data, min=0, max=1))
            
            print('Epoch: {:04d}'.format(epoch+1),
                        'loss_fro: {:.4f}'.format(loss_fro.item()),
                        'loss_feat: {:.4f}'.format(loss_smooth_feat.item()),
                        'loss_symmetric: {:.4f}'.format(loss_symmetric.item()),
                        'delta_l1_norm: {:.4f}'.format(torch.norm(estimator.estimated_adj-adj, 1).item()),
                        'loss_l1: {:.4f}'.format(loss_l1.item()),
                        'loss_total: {:.4f}'.format(total_loss.item()),
                        'loss_nuclear: {:.4f}'.format(loss_nuclear.item()))

            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            '''
            self.model.eval()
            normalized_adj = estimator.normalize()
            output = self.model(features, normalized_adj)

            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = accuracy(output[idx_val], labels[idx_val])
            print('Epoch: {:04d}'.format(epoch+1),
                'acc_train: {:.4f}'.format(acc_train.item()),
                'loss_val: {:.4f}'.format(loss_val.item()),
                'acc_val: {:.4f}'.format(acc_val.item()),
                'time: {:.4f}s'.format(time.time() - t))

            if acc_val > self.best_val_acc:
                self.best_val_acc = acc_val
                self.best_graph = normalized_adj.detach()
                self.weights = deepcopy(self.model.state_dict())
                if args.debug:
                    print(f'\t=== saving current graph/gcn, best_val_acc: %s' % self.best_val_acc.item())

            if loss_val < self.best_val_loss:
                self.best_val_loss = loss_val
                self.best_graph = normalized_adj.detach()
                self.weights = deepcopy(self.model.state_dict())
                if args.debug:
                    print(f'\t=== saving current graph/gcn, best_val_loss: %s' % self.best_val_loss.item())

            if args.debug:
                if epoch % 1 == 0:
                    print('Epoch: {:04d}'.format(epoch+1),
                        'loss_fro: {:.4f}'.format(loss_fro.item()),
                        'loss_gcn: {:.4f}'.format(loss_gcn.item()),
                        'loss_feat: {:.4f}'.format(loss_smooth_feat.item()),
                        'loss_symmetric: {:.4f}'.format(loss_symmetric.item()),
                        'delta_l1_norm: {:.4f}'.format(torch.norm(estimator.estimated_adj-adj, 1).item()),
                        'loss_l1: {:.4f}'.format(loss_l1.item()),
                        'loss_total: {:.4f}'.format(total_loss.item()),
                        'loss_nuclear: {:.4f}'.format(loss_nuclear.item()))
            '''
    
    
    
#feature_smoothingpart in Pro-GNN
    def feature_smoothing(self, adj, X):
        adj = (adj.t() + adj)/2
        rowsum = adj.sum(1)
        r_inv = rowsum.flatten()
        D = torch.diag(r_inv)
        L = D - adj

        r_inv = r_inv  + 1e-3
        r_inv = r_inv.pow(-1/2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        # L = r_mat_inv @ L
        L = r_mat_inv @ L @ r_mat_inv

        XLXT = torch.matmul(torch.matmul(X.t(), L), X)
        loss_smooth_feat = torch.trace(XLXT)
        return loss_smooth_feat
    
    
    def evaluate_adj_by_cls(self, Adj, features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, args):

        model = GCN(in_channels=nfeats, hidden_channels=args.hidden_dim_cls, out_channels=nclasses, num_layers=args.nlayers_cls,
                    dropout=args.dropout_cls, dropout_adj=args.dropedge_cls, Adj=Adj, sparse=args.sparse)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_cls, weight_decay=args.w_decay_cls)

        bad_counter = 0
        best_val = 0
        best_model = None

        if torch.cuda.is_available():
            model = model.cuda()
            train_mask = train_mask.cuda()
            val_mask = val_mask.cuda()
            test_mask = test_mask.cuda()
            features = features.cuda()
            labels = labels.cuda()

        for epoch in range(1, args.epochs_cls + 1):
            model.train()
            loss, accu = self.loss_cls(model, train_mask, features, labels)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            if epoch % 10 == 0:
                model.eval()
                val_loss, accu = self.loss_cls(model, val_mask, features, labels)
                if accu > best_val:
                    bad_counter = 0
                    best_val = accu
                    best_model = copy.deepcopy(model)
                else:
                    bad_counter += 1

                if bad_counter >= args.patience_cls:
                    break
        best_model.eval()
        test_loss, test_accu = self.loss_cls(best_model, test_mask, features, labels)
        return best_val, test_accu, best_model


    def train(self, args):

        torch.cuda.set_device(args.gpu)

        if args.gsl_mode == 'structure_refinement':
            features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, adj_original = load_data(args)
        elif args.gsl_mode == 'structure_inference':
            features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, _ = load_data(args)

        if args.downstream_task == 'classification':
            test_accuracies = []
            validation_accuracies = []
        elif args.downstream_task == 'clustering':
            n_clu_trials = copy.deepcopy(args.ntrials)
            args.ntrials = 1
            
        start = time.time()
        '''
        if args.gsl_mode == 'structure_inference':
            if args.sparse:
                anchor_adj_raw = torch_sparse_eye(features.shape[0])
            else:
                anchor_adj_raw = torch.eye(features.shape[0])
        elif args.gsl_mode == 'structure_refinement':
            if args.sparse:
                anchor_adj_raw = adj_original
            else:
                anchor_adj_raw = torch.from_numpy(adj_original)

        estimator = EstimateAdj(anchor_adj_raw, symmetric='sym', device=args.gpu).to(args.gpu)
        self.estimator = estimator
        '''
        

        for trial in range(args.ntrials):

            self.setup_seed(trial)
            

            if args.gsl_mode == 'structure_inference':
                if args.sparse:
                    anchor_adj_raw = torch_sparse_eye(features.shape[0])
                else:
                    anchor_adj_raw = torch.eye(features.shape[0])
            elif args.gsl_mode == 'structure_refinement':
                if args.sparse:
                    anchor_adj_raw = adj_original
                else:
                    anchor_adj_raw = torch.from_numpy(adj_original)

            estimator = EstimateAdj(anchor_adj_raw, symmetric='sym', device=args.gpu).to(args.gpu)
            
            self.estimator = estimator
            anchor_adj = normalize(anchor_adj_raw, 'sym', args.sparse)
            #anchor_adj = normalize(estimator.estimated_adj, 'sym', args.sparse)

            if args.sparse:
                anchor_adj_torch_sparse = copy.deepcopy(anchor_adj)
                anchor_adj = torch_sparse_to_dgl_graph(anchor_adj)

            if args.type_learner == 'fgp':
                graph_learner = FGP_learner(features.cpu(), args.k, args.sim_function, 6, args.sparse)
            elif args.type_learner == 'mlp':
                graph_learner = MLP_learner(2, features.shape[1], args.k, args.sim_function, 6, args.sparse,
                                     args.activation_learner)
            elif args.type_learner == 'att':
                graph_learner = ATT_learner(2, features.shape[1], args.k, args.sim_function, 6, args.sparse,
                                          args.activation_learner)
            elif args.type_learner == 'gnn':
                graph_learner = GNN_learner(2, features.shape[1], args.k, args.sim_function, 6, args.sparse,
                                     args.activation_learner, anchor_adj)

            model = GCL(nlayers=args.nlayers, in_dim=nfeats, hidden_dim=args.hidden_dim,
                         emb_dim=args.rep_dim, proj_dim=args.proj_dim,
                         dropout=args.dropout, dropout_adj=args.dropedge_rate, sparse=args.sparse)

            optimizer_cl = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_decay)
            optimizer_learner = torch.optim.Adam(graph_learner.parameters(), lr=args.lr, weight_decay=args.w_decay)
            self.optimizer_adj = torch.optim.SGD(estimator.parameters(),
                              momentum=0.9, lr=args.lr_adj)
            self.optimizer_l1 = PGD(estimator.parameters(),
                        proxs=[prox_operators.prox_l1],
                        lr=5e-4, alphas=[5e-4])
            self.optimizer_nuclear = PGD(estimator.parameters(),
                  proxs=[prox_operators.prox_nuclear],
                  lr=args.lr_adj, alphas=[args.beta])
            
            


            if torch.cuda.is_available():
                model = model.cuda()
                graph_learner = graph_learner.cuda()
                train_mask = train_mask.cuda()
                val_mask = val_mask.cuda()
                test_mask = test_mask.cuda()
                features = features.cuda()
                labels = labels.cuda()
                estimator = estimator.cuda()
                if not args.sparse:
                    anchor_adj = anchor_adj.cuda()

            if args.downstream_task == 'classification':
                best_val = 0
                best_val_test = 0
                best_epoch = 0
            

            for epoch in range(1, args.epochs + 1):

                model.train()
                graph_learner.train()
                estimator.train()
                #a = estimator.estimated_adj
               
                
                
                self.train_adj_bypgnn(epoch,features,estimator.estimated_adj,args)
                loss, Adj = self.loss_gcl(model, graph_learner, features, anchor_adj,estimator.estimated_adj,0.6)
                
                

                optimizer_cl.zero_grad()
                optimizer_learner.zero_grad()
                
                
                
                loss.backward()
                
                optimizer_cl.step()
                optimizer_learner.step()
                
                

                # Structure Bootstrapping
                if (1 - args.tau) and (args.c == 0 or epoch % args.c == 0):
                    if args.sparse:
                        learned_adj_torch_sparse = dgl_graph_to_torch_sparse(Adj)
                        anchor_adj_torch_sparse = anchor_adj_torch_sparse * args.tau \
                                                  + learned_adj_torch_sparse * (1 - args.tau)
                        anchor_adj = torch_sparse_to_dgl_graph(anchor_adj_torch_sparse)
                    else:
                        anchor_adj = anchor_adj * args.tau + Adj.detach() * (1 - args.tau)

                print("Epoch {:05d} | CL Loss {:.4f}".format(epoch, loss.item()))

                if epoch % args.eval_freq == 0:
                    if args.downstream_task == 'classification':
                        model.eval()
                        graph_learner.eval()
                        f_adj = Adj

                        if args.sparse:
                            f_adj.edata['w'] = f_adj.edata['w'].detach()
                        else:
                            f_adj = f_adj.detach()

                        val_accu, test_accu, _ = self.evaluate_adj_by_cls(f_adj, features, nfeats, labels,
                                                                               nclasses, train_mask, val_mask, test_mask, args)

                        if val_accu > best_val:
                            best_val = val_accu
                            best_val_test = test_accu
                            best_epoch = epoch

                    elif args.downstream_task == 'clustering':
                        model.eval()
                        graph_learner.eval()
                        _, embedding = model(features, Adj)
                        embedding = embedding.cpu().detach().numpy()

                        acc_mr, nmi_mr, f1_mr, ari_mr = [], [], [], []
                        for clu_trial in range(n_clu_trials):
                            kmeans = KMeans(n_clusters=nclasses, random_state=clu_trial).fit(embedding)
                            predict_labels = kmeans.predict(embedding)
                            cm_all = clustering_metrics(labels.cpu().numpy(), predict_labels)
                            acc_, nmi_, f1_, ari_ = cm_all.evaluationClusterModelFromLabel(print_results=False)
                            acc_mr.append(acc_)
                            nmi_mr.append(nmi_)
                            f1_mr.append(f1_)
                            ari_mr.append(ari_)

                        acc, nmi, f1, ari = np.mean(acc_mr), np.mean(nmi_mr), np.mean(f1_mr), np.mean(ari_mr)

            if args.downstream_task == 'classification':
                validation_accuracies.append(best_val.item())
                test_accuracies.append(best_val_test.item())
                print("Trial: ", trial + 1)
                print("Best val ACC: ", best_val.item())
                print("Best test ACC: ", best_val_test.item())
            elif args.downstream_task == 'clustering':
                print("Final ACC: ", acc)
                print("Final NMI: ", nmi)
                print("Final F-score: ", f1)
                print("Final ARI: ", ari)

        if args.downstream_task == 'classification' and trial != 0:
            self.print_results(validation_accuracies, test_accuracies)
        end = time.time()
        print("time:",end-start)


    def print_results(self, validation_accu, test_accu):
        s_val = "Val accuracy: {:.4f} +/- {:.4f}".format(np.mean(validation_accu), np.std(validation_accu))
        s_test = "Test accuracy: {:.4f} +/- {:.4f}".format(np.mean(test_accu),np.std(test_accu))
        print(s_val)
        print(s_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Experimental setting
    parser.add_argument('-dataset', type=str, default='cora',
                        choices=['cora', 'citeseer', 'pubmed'])
    parser.add_argument('-ntrials', type=int, default=5)
    parser.add_argument('-sparse', type=int, default=0)
    parser.add_argument('-gsl_mode', type=str, default="structure_inference",
                        choices=['structure_inference', 'structure_refinement'])
    parser.add_argument('-eval_freq', type=int, default=5)
    parser.add_argument('-downstream_task', type=str, default='classification',
                        choices=['classification', 'clustering'])
    parser.add_argument('-gpu', type=int, default=0)

    # GCL Module - Framework
    parser.add_argument('-epochs', type=int, default=1000)
    parser.add_argument('-lr', type=float, default=0.01)
    parser.add_argument('-w_decay', type=float, default=0.0)
    parser.add_argument('-hidden_dim', type=int, default=512)
    parser.add_argument('-rep_dim', type=int, default=64)
    parser.add_argument('-proj_dim', type=int, default=64)
    parser.add_argument('-dropout', type=float, default=0.5)
    parser.add_argument('-contrast_batch_size', type=int, default=0)
    parser.add_argument('-nlayers', type=int, default=2)

    # GCL Module -Augmentation
    parser.add_argument('-maskfeat_rate_learner', type=float, default=0.2)
    parser.add_argument('-maskfeat_rate_anchor', type=float, default=0.2)
    parser.add_argument('-dropedge_rate', type=float, default=0.5)

    # GSL Module
    parser.add_argument('-type_learner', type=str, default='fgp', choices=["fgp", "att", "mlp", "gnn"])
    parser.add_argument('-k', type=int, default=30)
    parser.add_argument('-sim_function', type=str, default='cosine', choices=['cosine', 'minkowski'])
    parser.add_argument('-gamma', type=float, default=0.9)
    parser.add_argument('-activation_learner', type=str, default='relu', choices=["relu", "tanh"])

    # Evaluation Network (Classification)
    parser.add_argument('-epochs_cls', type=int, default=200)
    parser.add_argument('-lr_cls', type=float, default=0.001)
    parser.add_argument('-w_decay_cls', type=float, default=0.0005)
    parser.add_argument('-hidden_dim_cls', type=int, default=32)
    parser.add_argument('-dropout_cls', type=float, default=0.5)
    parser.add_argument('-dropedge_cls', type=float, default=0.25)
    parser.add_argument('-nlayers_cls', type=int, default=2)
    parser.add_argument('-patience_cls', type=int, default=10)

    # Structure Bootstrapping
    parser.add_argument('-tau', type=float, default=1)
    parser.add_argument('-c', type=int, default=0)
    
    # Pro-GNN Parameters
    parser.add_argument('-lr_adj', type=float, default=0.01, help='lr for training adj')
    parser.add_argument('-alpha', type=float, default=5e-4, help='weight of l1 norm')
    parser.add_argument('-beta', type=float, default=1.5, help='weight of nuclear norm')
    
    parser.add_argument('-lambda_', type=float, default=0, help='weight of feature smoothing')
    parser.add_argument('-phi', type=float, default=0, help='weight of symmetric loss')
    parser.add_argument('-symmetric', action='store_true', default=False,
            help='whether use symmetric matrix')
    parser.add_argument('--debug', action='store_true',
        default=False, help='debug mode')

    args = parser.parse_args()

    experiment = Experiment()
    experiment.train(args)

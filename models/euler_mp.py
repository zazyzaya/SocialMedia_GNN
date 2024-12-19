import math 

import torch
from torch import nn
import torch.distributed.rpc as rpc
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch_geometric.nn.models import GraphSAGE

from loaders.topological_feats import topological_features

class RRefWrapper(DDP): 
    def call_fn(self, name, *args, **kwargs): 
        fn = getattr(self.module, name)
        return fn(*args, **kwargs)
    
    def train(self): 
        self.module.train()
    def eval(self): 
        self.module.eval()
    
class EulerGNN(nn.Module):
    def __init__(self, in_dim, hidden=64, gnn_layers=2):
        super().__init__()
        
        self.gnn = GraphSAGE(
            in_dim, hidden, 
            gnn_layers, dropout=0.2
        )
        
        self.xs = []
        self.eis = []
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self):
        embs = [
            self.gnn(self.xs[i], self.eis[i])
            for i in range(len(self.xs))
        ]
        return torch.stack(embs)
    
    def loss(self, zs): 
        # Only applies to first worker
        if zs.size(0) < len(self.eis): 
            eis = self.eis[1:]
        else: 
            eis = self.eis 

        pos = [
            (zs[i][eis[i][0]] * zs[i][eis[i][1]]).sum(dim=1)
            for i in range(len(eis))
        ]
        pos = torch.cat(pos)

        neg = []
        for i in range(len(eis)):
            rnd_src = torch.randint(0, eis[i][0].max(), eis[i].size(1))
            rnd_dst = torch.randint(0, eis[i][1].max(), eis[i].size(1))
            neg.append(
                (zs[i][rnd_src] * zs[i][rnd_dst]).sum(dim=1)
            )
        neg = torch.cat(neg)

        labels = torch.zeros(pos.size(0) + neg.size(0))
        labels[:pos.size(0)] = 1. 
        preds = torch.cat([pos,neg])

        loss = self.bce(preds, labels)
        return loss 
    
    def get_param_rrefs(self):
        rrefs = [rpc.RRef(param) for param in self.parameters()]
        return rrefs 
    
    def load_data(self, fname, idx_ptr): 
        '''
        Given filename of graph, and index pointer of where to split 
        edge index, load the segment of the graph this worker is 
        responsible for
        '''
        g = torch.load(fname, weights_only=False)
        
        self.eis = [
            g.edge_index[:, idx_ptr[i]:idx_ptr[i+1]]
            for i in range(idx_ptr.size(0)-1)
        ]
        self.xs = [
            topological_features(g.x, self.eis[i])
            for i in range(len(self.eis))
        ]


class Euler(nn.Module):
    def __init__(self, rrefs, hidden=64, rnn_layers=1, out_dim=32):
        super().__init__()

        self.workers = rrefs
        self.nworkers = len(rrefs)
        self.assigned = []

        self.rnn = nn.GRU(
            hidden, hidden,
            num_layers=rnn_layers,
            dropout=0.2 if rnn_layers > 1 else 0,
            batch_first=False
        )

        self.out = nn.Linear(hidden, out_dim)

    def forward(self, h0=None):
        '''
        No arguments, workers will have input data loaded
        in at inference/training time 
        '''
        xs = [
            self.workers[i].rpc_async().forward()
            for i in range(self.nworkers)
        ]
        zs = [x.wait() for x in xs]
        
        zs = torch.cat(zs, dim=0)
        zs, hn = self.rnn(zs, h0)
        return zs, hn 
    
    def loss(self, zs): 
        losses = []
        st = 1
        for i in range(self.nworkers): 
            en = self.assigned[i]
            losses.append(
                self.workers[i].rpc_async().loss(zs[st:en])
            )
            st = en 

        losses = [l.wait() for l in losses]
        return losses 
    
    def assign_work(self, fname, snapshot_idxs): 
        '''
        Makes each worker load the graph. Master 
        is responsible for load balancing the edges
        each worker recieves
        '''
        # TODO fix load imbalance
        jobs = len(snapshot_idxs)-1
        jobs_per_worker = math.ceil(jobs / self.nworkers)
        self.assigned = [0]

        # Tell each process to load in their portion of the dataset
        futs = []
        for i in range(self.nworkers): 
            st = i*jobs_per_worker
            en = st+jobs_per_worker+1
            futs.append(
                self.workers[i].rpc_async().load_data(
                    fname, snapshot_idxs[st:en]
                )
            )

            self.assigned.append(
                self.assigned[-1] + \
                len(snapshot_idxs[st:en])-1
            )
        
        # Trim off leading 0 
        self.assigned = self.assigned[1:]

        # Wait for all workers to load jobs
        [f.wait() for f in futs]
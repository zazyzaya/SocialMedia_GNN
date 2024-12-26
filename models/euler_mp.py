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

    # A bunch of wrapper functions so I don't have to use
    # call_fn every time
    def train(self, mode=True):
        self.module.train(mode=mode)
    def eval(self):
        self.module.eval()
    def loss(self, zs):
        return self.module.loss(zs)
    def predict(self, zs, unique=True, add_neg=True, no_grad=False):
        return self.module.predict(zs, unique=unique, add_neg=add_neg, no_grad=no_grad)
    def switch_mode(self, mode):
        return self.module.switch_mode(mode)
    def load_data(self, fname, idx_ptr, mode):
        return self.module.load_data(fname, idx_ptr, mode)

class EulerGNN(nn.Module):
    def __init__(self, i, in_dim, hidden=64, gnn_layers=2):
        super().__init__()

        self.data = dict()
        self.pid = i

        self.gnn = GraphSAGE(
            in_dim, hidden,
            gnn_layers, dropout=0.2
        )
        self.hidden = hidden

        self.xs = []
        self.eis = []
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, no_grad=False):
        with torch.set_grad_enabled(not no_grad):
            if len(self.xs) == 0:
                return None

            embs = [
                self.gnn(self.xs[i], self.eis[i])
                for i in range(len(self.xs))
            ]
            return torch.stack(embs)

    def predict(self, zs, unique=True, add_neg=True, no_grad=False):
        with torch.set_grad_enabled(not no_grad):

            # Only applies to first worker
            if zs.size(0) < len(self.eis):
                eis = self.eis[1:]
            else:
                eis = self.eis

            if unique:
                eis = [torch.unique(ei, dim=1) for ei in eis]

            # Added in for metric tracking
            troll_troll_edge = [
                self.xs[0][ei[0]][:, 0].logical_and(
                    self.xs[0][ei[1]][:, 0]
                )
                for ei in eis
            ]
            troll_troll_edge = torch.cat(troll_troll_edge)

            pos = [
                (zs[i][eis[i][0]] * zs[i][eis[i][1]]).sum(dim=1)
                for i in range(len(eis))
            ]
            pos = torch.cat(pos)

            # Negative sample if desired
            troll_troll_edge_neg = []
            if add_neg:
                neg = []
                for i in range(len(eis)):
                    rnd_src = torch.randint(0, eis[i][0].max(), (eis[i].size(1),))
                    rnd_dst = torch.randint(0, eis[i][1].max(), (eis[i].size(1),))

                    tten = self.xs[0][rnd_src][:, 0].logical_and(
                        self.xs[0][rnd_dst][:, 0]
                    )
                    troll_troll_edge_neg.append(tten)

                    neg.append(
                        (zs[i][rnd_src] * zs[i][rnd_dst]).sum(dim=1)
                    )
                neg = torch.cat(neg)
                troll_troll_edge_neg = torch.cat(troll_troll_edge_neg)
                return pos,neg, troll_troll_edge, troll_troll_edge_neg

            return pos, troll_troll_edge

    def loss(self, zs):
        pos,neg,_,_ = self.predict(zs, unique=False, add_neg=True)
        labels = torch.zeros(pos.size(0) + neg.size(0))
        labels[:pos.size(0)] = 1.
        preds = torch.cat([pos,neg])

        loss = self.bce(preds, labels)
        return loss

    def get_param_rrefs(self):
        rrefs = [rpc.RRef(param) for param in self.parameters()]
        return rrefs

    def switch_mode(self, mode):
        x,ei = self.data[mode]
        self.xs = x
        self.eis = ei
        return

    def load_data(self, fname, idx_ptr, mode):
        '''
        Given filename of graph, and index pointer of where to split
        edge index, load the segment of the graph this worker is
        responsible for
        '''
        # If data has been loaded before, keep it cached so we can switch quickly
        if mode in self.data:
            return self.switch_mode(mode)

        g = torch.load(fname, weights_only=False)
        eis = [
            g.edge_index[:, idx_ptr[i]:idx_ptr[i+1]]
            for i in range(len(idx_ptr)-1)
        ]
        xs = [
            topological_features(g.labels, eis[i])
            for i in range(len(eis))
        ]

        print(f"Worker {self.pid} assigned {len(eis)} graphs")

        self.data[mode] = (xs,eis)
        self.xs = xs
        self.eis = eis


class Euler(nn.Module):
    def __init__(self, rrefs, hidden=64, rnn_layers=1, out_dim=32):
        super().__init__()

        self.workers = rrefs
        self.nworkers = len(rrefs)
        self.assigned = dict()
        self.mode = None

        self.rnn = nn.GRU(
            hidden, hidden,
            num_layers=rnn_layers,
            dropout=0.2 if rnn_layers > 1 else 0,
            batch_first=False
        )

        self.out = nn.Linear(hidden, out_dim)

    def train(self, mode = True):
        [worker.rpc_sync().train(mode=mode) for worker in self.workers]
        return super().train(mode)
    def eval(self):
        [worker.rpc_sync().eval() for worker in self.workers]
        return super().eval()

    def parameter_rrefs(self):
        '''
        Distributed optimizer needs RRefs to params rather than the literal
        locations of them that you'd get with self.parameters(). This returns
        a parameter list of all remote workers and an RRef of the RNN held by
        the recurrent layer
        '''
        params = []
        for rref in self.workers:
            params.extend(
                rref.rpc_sync().call_fn('get_param_rrefs')
            )

        params.extend([rpc.RRef(param) for param in self.parameters()])
        return params

    def forward(self, h0=None, no_grad=False):
        '''
        No arguments, workers will have input data loaded
        in at inference/training time
        '''
        xs = [
            self.workers[i].rpc_async().forward(no_grad=no_grad)
            for i in range(self.nworkers)
        ]
        zs = [x.wait() for x in xs]
        zs = [z for z in zs if z is not None]

        zs = torch.cat(zs, dim=0)
        zs, hn = self.rnn(zs, h0)
        return zs, hn

    def loss(self, zs):
        losses = []
        for i in range(len(self.assigned[self.mode])-1):
            st = max(self.assigned[self.mode][i], 1)
            en = self.assigned[self.mode][i+1]

            if st != en:
                losses.append(
                    self.workers[i].rpc_async().loss(zs[st-1:en-1])
                )

        losses = [l.wait() for l in losses]
        return losses

    def assign_work(self, fname, snapshot_idxs, tag):
        '''
        Makes each worker load the graph. Master
        is responsible for load balancing the edges
        each worker recieves
        '''
        self.mode = tag
        if tag in self.assigned:
            futs = [
                self.workers[i].rpc_async().switch_mode(tag)
                for i in range(len(self.assigned[tag])-1)
            ]
            [f.wait() for f in futs]
            return

        # TODO fix load imbalance
        jobs = len(snapshot_idxs)-1
        jobs_per_worker = jobs // self.nworkers
        remainder = jobs % self.nworkers
        self.assigned[tag] = [0]

        # Tell each process to load in their portion of the dataset
        futs = []
        ranges = []
        st = 0
        for i in range(self.nworkers):
            en = st+jobs_per_worker

            if i < remainder:
                en += 1

            if st != en:
                ranges.append((st,en))
                futs.append(
                    self.workers[i].rpc_async().load_data(
                        fname, snapshot_idxs[st:en+1], tag
                    )
                )

                self.assigned[tag].append(
                    self.assigned[tag][-1] + \
                    len(snapshot_idxs[st:en])
                )
            else:
                break

            st = en

        #print(self.assigned)

        # Trim off leading 0
        #self.assigned[tag] = self.assigned[tag][1:]

        # Wait for all workers to load jobs
        [f.wait() for f in futs]

    def predict(self, zs, unique=True, add_neg=True, no_grad=False):
        jobs = []
        for i in range(len(self.assigned[self.mode])-1):
            st = max(self.assigned[self.mode][i], 1)
            en = self.assigned[self.mode][i+1]

            if st == en:
                continue

            jobs.append(
                self.workers[i].rpc_async().predict(
                    zs[st-1:en-1],
                    no_grad=no_grad,
                    unique=unique,
                    add_neg=add_neg
                )
            )

        jobs = [j.wait() for j in jobs]
        posses,negs,ttep,tten = zip(*jobs)

        pos = torch.cat(posses)
        neg = torch.cat(negs)
        ttep = torch.cat(ttep)
        tten = torch.cat(tten)

        return pos,neg,ttep,tten

    def save(self, fname):
        rnn = self.rnn.state_dict()
        gnn = self.workers[0].rpc_sync().call_fn('state_dict')

        torch.save({'rnn': rnn, 'gnn': gnn}, fname)
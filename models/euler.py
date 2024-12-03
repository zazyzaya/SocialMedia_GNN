import torch
from torch import nn
from torch.optim import Adam
from torch_geometric.nn.models import GraphSAGE

class Euler(nn.Module):
    def __init__(self, in_dim, hidden=64, gnn_layers=2, rnn_layers=1, out_dim=32, lr=0.001):
        super().__init__()

        self.gnn = GraphSAGE(
            in_dim, hidden,
            gnn_layers, dropout=0.2
        )

        self.rnn = nn.GRU(
            hidden, hidden,
            num_layers=rnn_layers,
            dropout=0.2,
            batch_first=False
        )

        self.out = nn.Linear(hidden, out_dim)

        self.opt = Adam(self.parameters(), lr=lr)
        self.bce = nn.BCEWithLogitsLoss()

    def embed(self, x, eis, h0=None):
        zs = [self.gnn(x, ei) for ei in eis]
        zs = torch.stack(zs, dim=0) # t x |V| x h
        zs,h = self.rnn(zs, h0)

        zs = self.out(zs)       # t x |V| x out
        return zs,h

    def predict(self, zs, eis, neg_ratio=0):
        pos = []
        neg = []

        for i,ei in enumerate(eis[1:]):
            pos.append(
                (zs[i][ei[0]] * zs[i][ei[1]]).sum(dim=1)
            )

            if neg_ratio:
                rnd_src = torch.randint(0, ei.max()+1, (int(ei.size(1)*neg_ratio),))
                rnd_dst = torch.randint(0, ei.max()+1, (int(ei.size(1)*neg_ratio),))
                neg.append(
                    (zs[i][rnd_src] * zs[i][rnd_dst]).sum(dim=1)
                )

        pos = torch.cat(pos)
        neg = torch.cat(neg)

        if neg_ratio:
            return pos,neg
        return pos


    def forward(self, x, eis, neg_ratio=1., predict=True):
        '''
        If predict is true, optimize P(e_{t+1} | z_t)
        Else, optimize P(e_t | z_t)
        '''
        self.train()
        self.opt.zero_grad()
        zs,h = self.embed(x, eis)

        if predict:
            pos,neg = self.predict(zs, eis, neg_ratio=neg_ratio)
        else:
            raise NotImplementedError("Link detection is not yet implemented")

        labels = torch.zeros(pos.size(0) + neg.size(0))
        labels[:pos.size(0)] = 1.

        loss = self.bce(torch.cat([pos,neg]), labels)
        loss.backward()
        self.opt.step()

        return loss.item(),h
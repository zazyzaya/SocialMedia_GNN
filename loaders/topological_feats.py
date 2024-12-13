import torch
from torch_geometric.nn import MessagePassing as MP
from torch_geometric.utils import to_undirected

def add_labels(g):
    '''
    Initially, normal users have no features, and trolls do
    '''
    new_x = torch.zeros(g.x.size(0), 2)

    is_user = (g.x.sum(dim=1) == 0).nonzero()
    is_troll = (g.x.sum(dim=1) != 0).nonzero()

    new_x[is_user, 0] = 1
    new_x[is_troll, 1] = 1

    return new_x

def add_topo_info(g):
    '''
    Assumes g.x contains only labels
    '''

    # Get src,dst, and bidirectional neighborhood info
    mp = MP(aggr=['sum', 'mean'])
    src = mp.propagate(g.edge_index, x=g.x)
    dst = mp.propagate(g.edge_index[[1,0]], x=g.x)
    ei = to_undirected(g.edge_index)
    bi = mp.propagate(ei, x=g.x)

    return torch.cat([src,dst,bi], dim=1)
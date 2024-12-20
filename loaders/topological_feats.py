import torch
from torch_geometric.nn import MessagePassing as MP
from torch_geometric.utils import to_undirected

def add_labels(g):
    '''
    Initially, normal users have no features, and trolls do
    '''
    x = g.x
    is_troll = (g.x.sum(dim=1) > 0)
    is_user = ~is_troll

    labels = torch.zeros(x.size(0), 2)
    labels[is_troll, 0] = 1
    labels[is_user, 1] = 1

    return labels

def topological_features(x, ei):
    '''
    Assumes x contains only labels
    '''

    # Get src,dst, and bidirectional neighborhood info
    mp = MP(aggr=['sum', 'mean'])
    src = mp.propagate(ei, x=x)
    dst = mp.propagate(ei[[1,0]], x=x)
    ei_ = to_undirected(ei)
    bi = mp.propagate(ei_, x=x)

    # Sum of neighbors (assumes labels are one-hot)
    degree = bi[:, list(range(x.size(1))) ].sum(dim=1, keepdim=True)
    degree_centrality = degree / degree.max()

    return torch.cat([x, src,dst,bi, degree_centrality], dim=1)
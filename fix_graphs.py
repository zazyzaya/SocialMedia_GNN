import glob
import torch
from loaders.utils import sort_edges

graphs = glob.glob('graphs/*.pt')
for gfile in graphs:
    g = torch.load(gfile, weights_only=False)
    '''
    g = sort_edges(g)
    torch.save(g, gfile)
    '''
    x = g.x
    is_troll = (g.x.sum(dim=1) > 0)
    is_user = ~is_troll

    labels = torch.zeros(x.size(0), 2)
    labels[is_troll, 0] = 1
    labels[is_user, 1] = 1
    g.labels = labels

    torch.save(g, gfile)
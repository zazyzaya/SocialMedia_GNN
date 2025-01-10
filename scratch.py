import torch
from loaders.utils import split_data_csr

g = torch.load('graphs/jan2019_russia-txt.pt')
split_data_csr(g)
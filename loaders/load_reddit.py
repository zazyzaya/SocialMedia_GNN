import torch
from torch_geometric.data import Data
from tqdm import tqdm

READ_DIR = '/mnt/raid1_ssd_4tb/datasets/reddit_disinfo'
NODES = 'reddit_node_labels.txt'
EDGES = 'reddit_graph_edges.txt'

def load_nodes():
    x = []
    node_map = dict()

    f = open(f'{READ_DIR}/{NODES}', 'r')
    line = f.readline()
    prog = tqdm(desc='Users')

    while line:
        line = line[:-1] # Trim newline
        usr,is_troll = line.split(' ')

        if usr not in node_map:
            node_map[usr] = len(node_map)
            x.append(int(is_troll))
        else:
            print("This shouldn't happen")

        line = f.readline()
        prog.update()

    f.close()
    prog.close()

    # Convert to one-hot
    xf = torch.tensor(x)
    x = torch.zeros(len(x), 2)
    x[xf == 1, 0] = 1.
    x[xf == 0, 1] = 1.

    return x, node_map

def load_edges(node_map):
    src,dst,ts = [],[]

    f = open(f'{READ_DIR}/{EDGES}', 'r')
    prog = tqdm(desc='Edges')

    line = f.readline()
    while line:
        line = line[:-1]
        s,d,t = line.split(' ')

        src.append(node_map[s])
        dst.append(node_map[d])
        ts.append(int(t))

        line = f.readline()
        prog.update()

    f.close()
    prog.close()

    ei = torch.tensor([src,dst])
    ts = torch.tensor(ts)

    ts,idx = ts.sort()
    ei = ei[:, idx]

    return ei, ts

def build_dataset():
    x,node_map = load_nodes()
    ei,ts = load_edges(node_map)

    return Data(
        x=x,
        edge_index=ei,
        ts=ts
    )

if __name__ == '__main__':
    g = build_dataset()
    torch.save(g, '../graphs/reddit_graph.pt')
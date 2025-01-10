import math
import torch

def sort_edges(g):
    ts,idx = g.ts.sort()

    g.ts = ts
    g.edge_index = g.edge_index[:, idx]
    g.edge_attr = g.edge_attr[idx]

    if 'text' in g.keys():
        g.text = [g.text[i] for i in idx]

    return g

def merge_unknown_users(g):
    unk = (g.x == 0).sum(dim=1) == g.x.size(1)
    unk = unk.nonzero()

    # Done in preprocessing now
    if unk.size(0) == 0:
        if g.edge_index.max() != g.size(0)-1:
            g.x = torch.cat([g.x, torch.zeros(1,g.x.size(1))])
        return g

    unk_id = unk.min()

    # Merge all normal users into single node
    g.x = g.x[:unk_id+1]
    g.edge_index[g.edge_index > unk_id] = unk_id

    return g

def split_data(g, snapshot_duration=(60*60*24*7)):
    '''
    Splits into snapshots of 1-month duration by default
    '''
    g.ts = g.ts - g.ts.min()
    tot_snapshots = math.ceil(g.ts.max() / snapshot_duration)

    eis = []
    min_ts = 0
    for _ in range(tot_snapshots):
        mask = (
            g.ts < min_ts + snapshot_duration
        ).logical_and(g.ts >= min_ts)

        ei = g.edge_index[:, mask]

        if ei.size(1):
            eis.append(ei)

        min_ts += snapshot_duration

    if eis == []:
        return [],[],[],0

    avg_size = sum([ei.size(1) for ei in eis]) / len(eis)
    print(f"{tot_snapshots} snapshots with sizes {[ei.size(1) for ei in eis]}")

    # 85 / 5 /10 split
    tr_idx = int(len(eis) * 0.85)
    te_idx = int(len(eis) * 0.90)

    tr = eis[:tr_idx]
    va = eis[tr_idx:te_idx]
    te = eis[te_idx:]

    return tr,va,te, avg_size

def split_data_csr(g, snapshot_duration=(60*60*24*7)):
    '''
    Splits into snapshots of 1-month duration by default
    Assumes ei's are sorted by time
    Returns
    '''
    g.ts = g.ts - g.ts.min()
    tot_snapshots = math.ceil(g.ts.max() / snapshot_duration)

    eis = [0]
    min_ts = 0
    sizes = []
    for i in range(tot_snapshots):
        mask = (g.ts < (min_ts + snapshot_duration)).nonzero()
        en = mask.max()
        min_ts += snapshot_duration

        # Don't add empty spans
        if en != eis[-1]:
            eis.append(en)
            sizes.append(eis[-1] - eis[-2])

    if eis == []:
        return [],[],[],0

    # 85 / 5 /10 split
    tr_idx = int(len(eis) * 0.85)
    te_idx = int(len(eis) * 0.90)

    tr = eis[:tr_idx+1]
    va = eis[tr_idx:te_idx+1]
    te = eis[te_idx:]

    return tr,va,te, sum(sizes)/len(sizes)
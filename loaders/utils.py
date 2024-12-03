import math
import torch

def merge_unknown_users(g):
    unk = (g.x == 0).sum(dim=1) == g.x.size(1)
    unk_id = unk.nonzero().min()

    # Merge all normal users into single node
    g.x = g.x[:unk_id+1]
    g.edge_index[g.edge_index > unk_id] = unk_id

    return g

def split_data(g, snapshot_duration=(60*60*24*30)):
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

    avg_size = sum([ei.size(1) for ei in eis]) / len(eis)
    print(f"{tot_snapshots} snapshots with sizes {[ei.size(1) for ei in eis]}")

    # 85 / 5 /10 split
    tr_idx = int(len(eis) * 0.85)
    te_idx = int(len(eis) * 0.90)

    tr = eis[:tr_idx]
    va = eis[tr_idx:te_idx]
    te = eis[te_idx:]

    return tr,va,te

import glob
import math
from types import SimpleNamespace

from joblib import Parallel, delayed
import torch
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score as auc_score,
    average_precision_score as ap_score
)

from loaders.utils import merge_unknown_users, split_data
from models.euler import Euler

HP = SimpleNamespace(
    epochs = 1000,
    lr = 0.001,
    patience = 25,
)

def train(model, args, kwargs, data, epochs, patience):
    torch.set_num_threads(64)
    model = model(*args, **kwargs)

    best = (-1,float('-inf'))
    log = []

    for e in range(epochs):
        loss,h = model(data.x, data.train)
        print(f"[{e}]\t{loss:0.4f}")

        with torch.no_grad():
            model.eval()

            # Get vaidation score
            val_z,h = model.embed(data.x, data.val, h0=h)

            # Don't inflate AUC/AP scores
            uq_val = [ei.unique(dim=1) for ei in data.val]
            val_p, val_n = model.predict(val_z, uq_val, neg_ratio=1)
            val_labels = torch.zeros(val_p.size(0) + val_n.size(0))
            val_labels[:val_p.size(0)] = 1
            val = torch.cat([val_p, val_n])

            val_auc = auc_score(val_labels, val)
            val_ap = ap_score(val_labels, val)
            print(f"\tV-AUC: {val_auc:0.4f}, V-AP: {val_ap:0.4f}", end='')

            if val_ap > best[1]:
                print("*")
            else:
                print()

            # Get test score
            test_z,_ = model.embed(data.x, data.test, h0=h)

            uq_test = [ei.unique(dim=1) for ei in data.test]
            test_p, test_n = model.predict(test_z, uq_test, neg_ratio=1)
            test_labels = torch.zeros(test_p.size(0) + test_n.size(0))
            test_labels[:test_p.size(0)] = 1
            test = torch.cat([test_p, test_n])

            test_auc = auc_score(test_labels, test)
            test_ap = ap_score(test_labels, test)
            print(f"\tT-AUC: {test_auc:0.4f}, T-AP: {test_ap:0.4f}")

        log.append([val_auc, val_ap, test_auc, test_ap])

        if val_ap > best[1]:
            best = (e, val_ap)
            model.save('weights/euler.pt')

        if e-best[0] == patience:
            print("Early stopping!")
            print("Best scores: ")
            print(log[best[0]])
            break

    return [best[0]] + log[best[0]]

def mean(l): return sum(l) / len(l)

def stderr(l):
    mu = mean(l)
    s = [math.pow(l_-mu, 2) for l_ in l]
    std = mean(s)
    return std / math.sqrt(len(l))

def compute_one(fname):
    g = torch.load(f'graphs/{fname}.pt')
    args = (g.x.size(1),)
    kwargs = dict(lr=HP.lr)

    g = merge_unknown_users(g)
    tr,va,te,avg_size = split_data(g)
    dur = 7
    potentials = [30, 90, 180, 365]
    i = 0
    while avg_size < 100:
        if i > len(potentials):
            print("Too small dataset")
            return

        if len(va) == 0:
            print("Too small dataset")
            return

        dur = potentials[i]
        tr,va,te,avg_size = split_data(
            g,
            snapshot_duration=(60 * 60 * 24 * dur)
        )
        i += 1

    # Not enough data
    if len(va) == 0:
        return

    data = SimpleNamespace(
        x=g.x, train=tr, val=va, test=te
    )

    results = [
        train(Euler, args, kwargs, data, HP.epochs, HP.patience)
        for _ in range(5)
    ]
    epoch, vauc, vap, tauc, tap = zip(*results)

    with open(f'results/{fname}.csv', 'w+') as f:
        f.write(f"Using {dur}-day snapshots\n")
        f.write('epoch,V-AUC,V-AP,T-AUC,T-AP\n')
        for i in range(len(vauc)):
            f.write(f'{epoch[i]},{vauc[i]},{vap[i]},{tauc[i]},{tap[i]}\n')

        f.write('\n')
        f.write(f'mean,{mean(vauc)},{mean(vap)},{mean(tauc)},{mean(tap)}\n')
        f.write(f'stderr,{stderr(vauc)},{stderr(vap)},{stderr(tauc)},{stderr(tap)}\n')

if __name__ == '__main__':
    files = glob.glob('graphs/*.pt')
    names = [f.split('/')[-1].replace('.pt','') for f in files]

    for name in names:
        compute_one(name)

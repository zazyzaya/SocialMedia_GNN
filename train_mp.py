import glob
import math
from types import SimpleNamespace

from joblib import Parallel, delayed
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.distributed.autograd as dist_autograd
from torch.distributed.optim import DistributedOptimizer
from torch.optim.adam import Adam
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
from sklearn.metrics import (
    roc_auc_score as auc_score,
    average_precision_score as ap_score
)

from loaders.utils import split_data_csr
from models.euler_mp import Euler, EulerGNN, RRefWrapper
from models.euler import Euler as Euler_Serial

WORLD_SIZE = 16
MAX_THREADS = 64
DIM = 15

import os
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '22032'

HP = SimpleNamespace(
    epochs = 1,
    lr = 0.001,
    patience = 25,
    hidden = 64,
    gnn_layers = 2
)

def _get_worker(pid, args=(), kwargs=dict()):
    print(f"Getting worker {pid}")
    return rpc.remote(
        f'worker{pid}',
        RRefWrapper,
        args = (EulerGNN(pid, *args, **kwargs),)
    )

def init_procs(rank, world_size, data):
    torch.set_num_threads(MAX_THREADS // world_size)

    # Required because gemini uses Infiniband(?)
    options = rpc.TensorPipeRpcBackendOptions(_transports=["uv"])

    if rank == world_size-1: # Master
        print(f"Initializing Master")
        rpc.init_rpc(
            'master',
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )

        # Build GNN embedders on workers
        rrefs = [
            _get_worker(
                i,
                args=(DIM,),
                kwargs=dict(hidden=HP.hidden, gnn_layers=HP.gnn_layers)
            ) for i in range(world_size-1)
        ]

        # Build model on master
        model = Euler(rrefs)

        # Send to training
        rets = train(model,data)

        # Flush to tmp file so parent process can get ret vals
        with open('tmp.txt', 'w+') as f:
            out = ','.join(rets)
            f.write(out)

    else:  # Init workers
        # So DDP knows they're all sharing a model
        print(f"Initializing proc group for worker{rank}")
        dist.init_process_group(
            'gloo', rank=rank, world_size=world_size-1
        )

        print(f"Initializing RPC for worker{rank}")

        rpc.init_rpc(
            f'worker{rank}',
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )

        print("Worker initialized")

    rpc.shutdown()

def train(model: Euler, data):
    best = (-1,float('-inf'))
    log = []

    opt = DistributedOptimizer(
        Adam,
        model.parameter_rrefs(),
        lr=0.01
    )

    for e in range(HP.epochs):
        model.assign_work(data.fname, data.train, 'tr')

        with dist_autograd.context() as cid:
            model.train()
            zs,h = model.forward()
            loss = model.loss(zs)
            dist_autograd.backward(cid, loss)
            opt.step(cid)

        loss = (sum(loss) / model.nworkers).item()
        print(f"[{e}] Loss:\t{loss:0.4f}")

        with torch.no_grad():
            model.eval()

            # Get vaidation score
            model.assign_work(data.fname, data.val, 'va')
            val_z,h = model.forward(h, no_grad=True)
            val_p, val_n, ttep, tten = model.predict(val_z, no_grad=True)
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
            model.assign_work(data.fname, data.test, 'te')
            test_z,_ = model.forward(h, no_grad=True)
            test_p, test_n, ttep, tten = model.predict(test_z, no_grad=True)
            test_labels = torch.zeros(test_p.size(0) + test_n.size(0))
            test_labels[:test_p.size(0)] = 1

            test = torch.cat([test_p, test_n])
            is_tte = torch.cat([ttep, tten])

            test_auc_tte = auc_score(test_labels[is_tte], test[is_tte])
            test_auc_tue = auc_score(test_labels[~is_tte], test[~is_tte])
            test_auc = auc_score(test_labels, test)

            test_ap_tte = ap_score(test_labels[is_tte], test[is_tte])
            test_ap_tue = ap_score(test_labels[~is_tte], test[~is_tte])
            test_ap = ap_score(test_labels, test)
            print(f"\tT-AUC: {test_auc:0.4f}, T-AP: {test_ap:0.4f}")
            print(f"\tTTE-AUC: {test_auc_tte:0.4f}, TTE-AP: {test_ap_tte:0.4f}")
            print(f"\tTUE-AUC: {test_auc_tue:0.4f}, TUE-AP: {test_ap_tue:0.4f}")

        log.append([val_auc, val_ap, test_auc, test_ap, test_auc_tte, test_ap_tte, test_auc_tue, test_ap_tue])

        if val_ap > best[1]:
            best = (e, val_ap)
            model.save('weights/euler_mp.pt')

        if e-best[0] == HP.patience:
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

def spinup(data, worldsize):
    mp.spawn(
        init_procs,
        args=(worldsize,data),
        nprocs=worldsize,
        join=True
    )
    with open('tmp.txt', 'r') as f:
        line = f.read()
        epoch, vauc, vap, tauc, tap, t_auc_tte, t_ap_tte, t_auc_tue, t_ap_tue = line.split(',')

    return int(epoch), float(vauc), float(vap), float(tauc), float(tap), \
        float(t_auc_tte), float(t_ap_tte), float(t_auc_tue), float(t_ap_tue)

def compute_one(fname):
    g = torch.load(f'graphs/{fname}.pt', weights_only=False)

    tr,va,te, avg_size = split_data_csr(g)
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
        tr,va,te,avg_size = split_data_csr(
            g,
            snapshot_duration=(60 * 60 * 24 * dur)
        )
        i += 1

    # Not enough data
    if len(va) == 0:
        return

    data = SimpleNamespace(
        x=g.labels, train=tr, val=va, test=te, fname=f'graphs/{fname}.pt'
    )

    results = [
        spinup(data, min(len(tr), WORLD_SIZE))
        for _ in range(5)
    ]
    epoch, vauc, vap, tauc, tap, t_auc_tte, t_ap_tte, t_auc_tue, t_ap_tue = zip(*results)

    with open(f'results/{fname}.csv', 'w+') as f:
        f.write(f"Using {dur}-day snapshots\n")
        f.write('epoch,V-AUC,V-AP,T-AUC,T-AP,TTE-AUC,TTE-AP,TUE-AUC,TUE-AP\n')
        for i in range(len(vauc)):
            f.write(f'{epoch[i]},{vauc[i]},{vap[i]},{tauc[i]},{tap[i]},{t_auc_tte[i]},{t_ap_tte[i]},{t_auc_tue[i]},{t_ap_tue[i]}\n')

        f.write('\n')
        f.write(f'mean,{mean(vauc)},{mean(vap)},{mean(tauc)},{mean(tap)},{mean(t_auc_tte)},{mean(t_ap_tte)},{mean(t_auc_tue)},{mean(t_ap_tue)}\n')
        f.write(f'stderr,{stderr(vauc)},{stderr(vap)},{stderr(tauc)},{stderr(tap)}\n')



if __name__ == '__main__':
    files = [
        'jan2019_iran',
        'jan2019_russia',
        #'jan2019/venezuela', # Crashed when building. Rerun
        'aug2019_china',
        'sept2019_uae'
    ]
    names = [f.split('/')[-1].replace('.pt','') for f in files]

    for name in names:
        compute_one(name)

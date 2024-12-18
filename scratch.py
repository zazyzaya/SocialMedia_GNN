import os

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp

WORLD_SIZE = 5

import os
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '22032'

class Worker(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Linear(in_dim,1)

    def forward(self, x):
        print("Worker, forward!")
        return self.net(x)

class Master(nn.Module):
    def __init__(self, rrefs):
        super().__init__()

        self.nets = rrefs
        self.nworkers = len(rrefs)

    def forward(self, x):
        print("Master, forward!")
        xs = [
            self.nets[i].rpc_async().forward(x)
            for i in range(self.nworkers)
        ]
        xs = [x.wait() for x in xs]
        return torch.cat(xs)

def get_worker(pid, args):
    print(f"Getting worker {pid}")
    return rpc.remote(
        f'worker{pid}',
        DDP,
        args = (Worker(*args),)
    )

def train(model):
    x = torch.rand(2,10)
    print(model(x))

def init_procs(rank, world_size):
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

        rrefs = [get_worker(i, (10,)) for i in range(world_size-1)]

        x = torch.rand(2,10)
        xs = [
            rref.rpc_async().forward(x)
            for rref in rrefs
        ]

        # Should all be the same
        out = torch.cat([x.wait() for x in xs], dim=1)
        print(out)

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

if __name__ == '__main__':
    mp.spawn(
        init_procs,
        args=(WORLD_SIZE,),
        nprocs=WORLD_SIZE,
        join=True
    )
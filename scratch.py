import torch 
from torch import nn 
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist 
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp 

class Worker(nn.Module): 
    def __init__(self, in_dim):
        self.net = nn.Linear(in_dim,1)
    def forward(self, x):
        return self.net(x) 

class Master(nn.Module): 
    def __init__(self, rrefs): 
        self.nets = rrefs 
        self.nworkers = len(rrefs)

    def forward(self, x): 
        xs = [
            self.nets[i].rpc_async().forward(x)
            for i in range(self.nworkers)
        ]
        xs = [x.wait() for x in xs]
        return torch.cat(xs)

def get_worker(pid, args): 
    return rpc.remote(
        f'worker{pid}', 
        DDP, 
        args = Worker(*args)
    )

def train(model):
    x = torch.rand(2,10)
    print(model(x))

def init_procs(rank, world_size): 
    print(f"Initializing process {rank}")
    if rank == world_size-1: # Master 
        rpc.init_rpc('master', rank=rank, world_size=world_size)

        rrefs = [get_worker(i, 10) for i in range(world_size-1)]
        model = Master(rrefs)
        train(model)

    else: 
        # So DDP knows they're all sharing a model
        dist.init_process_group(
            'gloo', rank=rank, world_size=world_size
        )

        # Init workers 
        rpc.init_rpc(f'worker{rank}', rank=rank, world_size=world_size)

if __name__ == '__main__':
    mp.spawn(
        init_procs, 
        nprocs=5, 
        join=True
    )
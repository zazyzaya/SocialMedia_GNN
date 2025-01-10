import os

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.distributed.autograd as dist_autograd
from torch.distributed.optim import DistributedOptimizer
from torch.optim.adam import Adam
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp

WORLD_SIZE = 5

import os
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '22032'


class MyDDP(DDP):
    def call_fn(self, name, *args, **kwargs):
        fn = getattr(self.module, name)
        return fn(*args, **kwargs)

    def get_module(self):
        return self.module

class Worker(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Linear(in_dim,1)
        self.mse = nn.MSELoss()

    def forward(self, x):
        print("Worker, forward!")
        return self.net(x)

    def loss(self, y_hat, y):
        return self.mse(y_hat,y)

    def get_param_rrefs(self):
        rrefs = [rpc.RRef(param) for param in self.parameters()]
        return rrefs

class Master(nn.Module):
    def __init__(self, rrefs):
        super().__init__()

        self.nets = rrefs
        self.nworkers = len(rrefs)

    def forward(self, x):
        print("Master, forward!")
        xs = [
            self.nets[i].rpc_async().forward(x[i])
            for i in range(self.nworkers)
        ]
        xs = [x.wait() for x in xs]
        return xs

    def loss(self, y_hat, y):
        losses = [
            self.nets[i].rpc_async().call_fn('loss', y_hat[i], y)
            for i in range(self.nworkers)
        ]
        losses = [loss.wait() for loss in losses]
        return losses

    def parameter_rrefs(self):
        '''
        Distributed optimizer needs RRefs to params rather than the literal
        locations of them that you'd get with self.parameters(). This returns
        a parameter list of all remote workers and an RRef of the RNN held by
        the recurrent layer
        '''
        params = []
        for rref in self.nets:
            params.extend(
                rref.rpc_sync().call_fn('get_param_rrefs')
            )

        #params.extend(_param_rrefs(self.rnn))
        return params

def get_worker(pid, args):
    print(f"Getting worker {pid}")
    return rpc.remote(
        f'worker{pid}',
        MyDDP,
        args = (Worker(*args),)
    )

def train(model):
    opt = DistributedOptimizer(
        Adam,
        model.parameter_rrefs(),
        lr=0.01
    )

    for _ in range(10):
        model.train()
        x = [torch.rand(10,10) for _ in range(WORLD_SIZE-1)]

        with dist_autograd.context() as cid:
            ret = model(x)
            labels = torch.zeros(ret[0].size())
            loss = model.loss(ret, labels)
            print(loss)
            dist_autograd.backward(cid, loss)
            opt.step(cid)


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
        model = Master(rrefs)
        train(model)

    else:  # Init workers
        # So DDP knows they're all sharing a model
        print(f"Initializing proc group for worker{rank}")
        dist.init_process_group(
            rank=rank, world_size=world_size-1
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
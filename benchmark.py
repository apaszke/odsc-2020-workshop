import time
from contextlib import contextmanager
from torch.autograd.profiler import record_function, profile

import torch
from model import TransformerModel, RNNModel


@contextmanager
def measure():
    s = time.perf_counter()
    try:
        yield
    finally:
        print(time.perf_counter() - s)


@torch.no_grad()
def main():
    model = TransformerModel(ntoken=100, ninp=8000, nhead=8,
                             nhid=10000, nlayers=1).to('cuda')
    time_steps = 64
    batch_size = 128
    input = torch.zeros(time_steps, batch_size, dtype=torch.int64, device='cuda')
    output = model(input)
    torch.cuda.synchronize()
    with measure():
        for i in range(4):
            output = model(input)
        torch.cuda.synchronize()


if __name__ == '__main__':
    main()

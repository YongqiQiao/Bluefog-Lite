import argparse
import copy
import os
import tqdm
import numpy as np

from torchvision import datasets, transforms
import torch.utils.data.distributed
import torch.nn.functional as F

import bluefoglite.torch_api as bfl
from bluefoglite.utility import broadcast_parameters
from bluefoglite.common import topology
from bluefoglite.common.torch_backend import ReduceOp
from model import MLP,CNN_MNIST

# Args
parser = argparse.ArgumentParser(
    description="Bluefog-Lite Example on MNIST",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--batch_size", type=int, default=128, help="input batch size for training"
)
parser.add_argument(
    "--test_batch_size", type=int, default=128, help="input batch size for testing"
)
parser.add_argument("--epochs", type=int, default=2000, help="number of epochs to train")
parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
parser.add_argument(
    "--log_interval",
    type=int,
    default=100,
    help="how many batches to wait before logging training status",
)
parser.add_argument(
    "--no_cuda", action="store_true", default=False, help="disables CUDA training"
)
parser.add_argument(
    "--seed", type=int, default=42, metavar="S", help="random seed (default: 42)"
)
parser.add_argument(
    "--profiling",
    type=str,
    default="c_profiling",
    metavar="S",
    help="enable which profiling? default: no",
    choices=["no_profiling", "c_profiling", "torch_profiling"],
)
parser.add_argument(
    "--disable-dynamic-topology",
    action="store_true",
    default=True,
    help="Disable each iteration to transmit one neighbor per iteration dynamically.",
)
parser.add_argument(
    "--backend",
    type=str,
    default="nccl",
    choices=["gloo", "nccl"],
)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Initialize topology
bfl.init(backend=args.backend)
topo = topology.RingGraph(bfl.size())
bfl.set_topology(topo)

# Device
if args.cuda:
    print("using cuda.")
    device_id = bfl.rank() % torch.cuda.device_count()
    torch.cuda.set_device(device_id)
    torch.cuda.manual_seed(args.seed)
else:
    print("using cpu")
    torch.manual_seed(args.seed)

# Dataloader
kwargs = {}
data_folder_loc = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

train_dataset = datasets.MNIST(
    os.path.join(data_folder_loc, "data", "data-%d" % bfl.rank()),
    train=True,
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    ),
)
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset, num_replicas=bfl.size(), rank=bfl.rank()
)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, sampler=train_sampler, **kwargs
)

test_dataset = datasets.MNIST(
    os.path.join(data_folder_loc, "data", "data-%d" % bfl.rank()),
    train=False,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    ),
)
test_sampler = torch.utils.data.distributed.DistributedSampler(
    test_dataset, num_replicas=bfl.size(), rank=bfl.rank()
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.test_batch_size, sampler=test_sampler, **kwargs
)

# model
model = CNN_MNIST()
if args.cuda:
    model.cuda()

losses = []
accuracies = []
# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

# Broadcast parameters & optimizer state
broadcast_parameters(model.state_dict(), root_rank=0)


def metric_average(val):
    tensor = torch.tensor(val)
    avg_tensor = bfl.allreduce(tensor)
    return avg_tensor.item()


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward() 
        optimizer.step()
        with torch.no_grad():
            # TODO[1]: Implement unit test to check whether params in different workers are same after allreduce
            # TODO[2]: Write a function to sychronize the parameters in different workers
            for module in model.parameters():
                bfl.allreduce(module.data, op=ReduceOp.AVG, inplace=True)
        losses.append(loss.item())
        accuracies.append(batch_idx / len(train_loader))
        if batch_idx % args.log_interval == 0:
            print(
                "[{}] Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t".format(
                    bfl.rank(),
                    epoch,
                    batch_idx * len(data),
                    len(train_sampler),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def test(epoch):
    model.eval()
    test_loss, test_accuracy, total = 0.0, 0.0, 0.0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]
        test_accuracy += pred.eq(target.data.view_as(pred)).cpu().float().sum().item()
        test_loss += F.cross_entropy(output, target, reduction="sum").item()
        total += len(target)
    test_loss /= total
    test_accuracy /= total
    # Bluefog: average metric values across workers.
    test_loss = metric_average(test_loss)
    test_accuracy = metric_average(test_accuracy)
    if bfl.rank() == 0:
        print(
            "\nTest Epoch: {}\tAverage loss: {:.6f}\tAccuracy: {:.4f}%\n".format(
                epoch, test_loss, 100.0 * test_accuracy
            ),
            flush=True,
        )


if args.profiling == "c_profiling":
    if bfl.rank() == 0:
        import cProfile
        import pstats

        profiler = cProfile.Profile()
        profiler.enable()
        for e in range(args.epochs):
            train(e)
            test(e)
        profiler.disable()
        # redirect to ./output_static.txt or ./output_dynamic.txt
        with open(
            f"/home/qyq/bfl/results/dgd_{'static' if args.disable_dynamic_topology else 'dynamic'}_np{bfl.size()}_{args.topology}.txt",
            "w",
        ) as file:
            stats = pstats.Stats(profiler, stream=file).sort_stats("tottime")
            stats.print_stats()
    else:
        for e in range(args.epochs):
            train(e)
            test(e)
elif args.profiling == "torch_profiling":
    from torch.profiler import profile, ProfilerActivity
    import contextlib

    assert args.backend != "nccl", "NCCL backend does not support torch_profiling."

    if bfl.rank() == 0:
        with profile(
            activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], record_shapes=True
        ) as prof:
            train(0)
        # redirect to ./output_static.txt or ./output_dynamic.txt
        with open(
            f"output/tp_{'static' if args.disable_dynamic_topology else 'dynamic'}_np{bfl.size()}_{args.topology}.txt",
            "w",
        ) as file:
            with contextlib.redirect_stdout(file):
                print(prof.key_averages().table(sort_by="cpu_time_total"))
    else:
        train(0)
else:
    for e in range(args.epochs):
        train(e)
        test(e)
bfl.barrier()
print(f"rank {bfl.rank()} finished.")

if bfl.rank()==0:
    f=open("/home/qyq/bfl/results/{}_loss.txt".format(dgd),'w')
    for loss in losses:
        f.write(str(loss)+'\n')
    f.close()
    f=open("/home/qyq/bfl/results/{}_accuracy.txt".format(dgd),'w')
    for accuracy in accuracies:
        f.write(str(accuracy)+'\n')
    f.close()

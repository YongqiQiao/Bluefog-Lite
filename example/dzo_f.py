import argparse
import os
import time
from torchvision import datasets, transforms
import torch.utils.data.distributed
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import torch
from copy import deepcopy
import bluefoglite
import bluefoglite.torch_api as bfl
from bluefoglite.common import topology
from bluefoglite.utility import (
    broadcast_parameters,
    broadcast_optimizer_state,
)
from bluefoglite.common.optimizers import (
    DistributedAdaptWithCombineOptimizer,
    DistributedGradientAllreduceOptimizer,
    CommunicationType,
)
from model import ResNet20, ResNet32, ResNet44, ResNet56, ViT, CNN_MNIST

# Args
parser = argparse.ArgumentParser(
    description="Bluefog-Lite Example on MNIST",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--model", type=str, default="cnn", help="model to use")
parser.add_argument(
    "--batch-size", type=int, default=64, help="input batch size for training"
)
parser.add_argument(
    "--test-batch-size", type=int, default=64, help="input batch size for testing"
)
parser.add_argument("--epochs", type=int, default=1000, help="number of epochs to train")
parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
parser.add_argument("--num-perturb", type=int, default=40, help="number of perturbations")
parser.add_argument("--zo-eps", type=float, default=0.0001, help="zo-eps")
parser.add_argument("--weight-decay", type=float, default=1e-5, help="weight-decay")
parser.add_argument("--experiment-name", type=str, default="", help="experiment-name")
parser.add_argument(
    "--communicate-state-dict",
    action="store_true",
    default=False,
    help="If True, communicate state dictionary of model instead of named parameters",
)
parser.add_argument(
    "--log-interval",
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
    "--backend",
    type=str,
    default="nccl",
    choices=["gloo", "nccl"],
)
parser.add_argument(
    "--profiling",
    type=str,
    default="no_profiling",
    metavar="S",
    help="enable which profiling? default: no",
    choices=["no_profiling", "c_profiling", "torch_profiling"],
)
parser.add_argument(
    "--topology",
    type=str,
    default="ring",
    help="The type of topology to use",
    choices=["ring", "exp", "exp2", "mesh2d", "star", "fully_connected"],
)
parser.add_argument(
    "--disable-dynamic-topology",
    action="store_true",
    default=True,
    help="Disable each iteration to transmit one neighbor per iteration dynamically.",
)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Initialize topology
bfl.init(backend=args.backend)
if args.topology == "ring":
    topo = topology.RingGraph(bfl.size())
elif args.topology == "exp":
    topo = topology.ExponentialGraph(bfl.size())
elif args.topology == "exp2":
    topo = topology.ExponentialTwoGraph(bfl.size())
elif args.topology == "mesh2d":
    topo = topology.MeshGrid2DGraph(bfl.size())
elif args.topology == "star":
    topo = topology.StarGraph(bfl.size())
else:
    raise NotImplementedError("topology not implemented")
bfl.set_topology(topo)
if not args.disable_dynamic_topology:
    dynamic_neighbor_allreduce_gen = topology.GetDynamicOnePeerSendRecvRanks(
        bfl.load_topology(), bfl.rank()
    )

# Device
if args.cuda:
    print("using cuda.")
    device_id = bfl.rank() % torch.cuda.device_count()
    torch.cuda.set_device(device_id)
    torch.cuda.manual_seed(args.seed)
    device = torch.tensor([0.0]).cuda().device
else:
    print("using cpu")
    torch.manual_seed(args.seed)
    device = "cpu"
    
# Dataloader
kwargs = {}
data_folder_loc = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
train_dataset = datasets.MNIST(
    root="/home/qyq/bfl/data", train=True, download=True, transform=transform
)
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset, num_replicas=bfl.size(), rank=bfl.rank(), seed=args.seed
)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    sampler=train_sampler,
    **kwargs,
)
test_dataset = datasets.MNIST(
    root="/home/qyq/bfl/data", train=False, download=True, transform=transform
)
test_sampler = torch.utils.data.distributed.DistributedSampler(
    test_dataset, num_replicas=bfl.size(), rank=bfl.rank(), seed=args.seed
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=args.test_batch_size,
    sampler=test_sampler,
    **kwargs,
)

model = CNN_MNIST()
if args.cuda:
    model.cuda()

named_parameters_to_optim = []
for name,param in model.named_parameters():
    if param.requires_grad:
        named_parameters_to_optim.append((name, param))
        param.grad=None

broadcast_parameters(model.state_dict(), root_rank=0)

lr = args.lr
epochs = args.epochs
num_perturb = args.num_perturb
zo_eps = args.zo_eps
weight_decay= args.weight_decay
projected_grad_list = [0]*num_perturb

topo = bfl.load_topology()
self_weight, recv_weights = bluefoglite.GetRecvWeights(topo, bfl.rank())
recv_weights = list(recv_weights.items())
neighbor_size = len(recv_weights)
neighbor_models = {}

losses = []
accuracies = []
for idx in range(neighbor_size):
    neighbor_models[idx] = deepcopy(named_parameters_to_optim)
grad_buffer = [torch.zeros(num_perturb) for _ in range(neighbor_size)]

def zo_perturb_parameters(scaling_factor, random_seed):
        """
        Perturb the parameters with random vector z.
        Input:
        - random_seed: random seed for MeZO in-place perturbation (if it's None, we will use self.zo_random_seed)
        - scaling_factor: theta = theta + scaling_factor * z * eps
        """

        # Set the random seed to ensure that we sample the same z for perturbation/update
        torch.manual_seed(random_seed)
        for _, param in named_parameters_to_optim:
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            param.data = param.data + scaling_factor * z * zo_eps

def zo_forward(inputs, targets):
    model.eval()
    outputs = model(inputs)
    loss = F.cross_entropy(outputs, targets)
    return loss.detach()

@torch.no_grad()
def zo_step(inputs, targets,epoch):
    for i in range(num_perturb):
        # Sample the random seed for sampling z
        # zo_random_seed = np.random.randint(1000000000)
        zo_random_seed = (epoch+1)*(i+1)
        loss = zo_forward(inputs, targets)
        # First function evaluation
        zo_perturb_parameters(scaling_factor=1, random_seed=zo_random_seed)
        loss1 = zo_forward(inputs, targets)

        projected_grad = ((loss1-loss) / (zo_eps)).item()
        projected_grad_list[i] = projected_grad
        # huanyuan
        zo_perturb_parameters(scaling_factor=-1, random_seed=zo_random_seed)
    return loss1

def zo_update(epoch):
    """
    Update the parameters with the estimated gradients.
    """
    grad_tmp = defaultdict(float)
    for param_idx, (name,param) in enumerate(named_parameters_to_optim):
        param.data = self_weight * param.data
        for neighbor_idx,(neighbor,recv_weight) in enumerate(recv_weights):
            # if bfl.rank()==0:
            #     print("param_name:",name,"neighbor_param_name:",neighbor_models[neighbor_idx][param_idx][0])
            #     print("param:",param.data[0],"neighbor_param:",neighbor_models[neighbor_idx][param_idx][1].data[0])
            param.data += recv_weight * neighbor_models[neighbor_idx][param_idx][1].data
        # if bfl.rank()==0:
        #     print("param:",param.data[0])
        # input()
    # Reset the random seed for sampling zs:
    for i in range(num_perturb):
        torch.manual_seed((epoch+1)*(i+1))
        for name, param in named_parameters_to_optim:
            # Resample z
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device,
                            dtype=param.data.dtype)
            grad_tmp[name] += projected_grad_list[i] * z
        # print("time of resample z is {}".format(end-start))
    for name, param in named_parameters_to_optim:
        if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
            param.data = param.data - lr * (
                grad_tmp[name] / num_perturb + weight_decay * param.data)
        else:
            param.data = param.data - lr * (grad_tmp[name] / num_perturb)
        # param.data = param.data - lr * (grad_tmp[name] / num_perturb)
    # print("time of zo_update is {}".format(end-start))

def zo_update_neighbor_models(epoch):
    for grad_idx, grads in enumerate(grad_buffer):
        grad_tmp=defaultdict(float)
        for i in range(num_perturb):
            torch.manual_seed((epoch+1)*(i+1))
            for name, param in neighbor_models[grad_idx]:
                # Resample z
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device,
                                dtype=param.data.dtype)
                grad_tmp[name] += grads[i] * z
        for name, param in neighbor_models[grad_idx]:
            if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                param.data = param.data - lr * (
                    grad_tmp[name] / num_perturb + weight_decay * param.data)
            else:
                param.data = param.data - lr * (grad_tmp[name] / num_perturb)

def metric_average(val):
    tensor = torch.tensor(val, device=device)
    avg_tensor = bfl.allreduce(tensor)
    return avg_tensor.item()  

def train(epoch):
    train_loss, correct, total = 0, 0, 0
    for batch_idx, (data,targets) in enumerate(train_loader):
        data, targets = data.cuda(), targets.cuda()
        loss = zo_step(data, targets, epoch)
        grad_tensor = torch.tensor(projected_grad_list, device=device)
        # print("rank:",bfl.rank(),"grad_tensor:",grad_tensor)
        grad_tensor = bfl.neighbor_allreduce(grad_tensor)
        # print("rank:",bfl.rank(),"avg_grad_tensor:",grad_tensor)
        zo_update(epoch)
        bfl.neighbor_allgather(grad_buffer, grad_tensor)
        # print("rank:",bfl.rank(),"grad_buffer:",grad_buffer)
        zo_update_neighbor_models(epoch)
        train_loss += loss.item()
        outputs = model(data)
        _, pred = outputs.max(dim=1)
        total += targets.size(dim=0)
        correct += pred.eq(targets).sum().item()

        if (batch_idx + 1) % 100 == 0:
            print(
                "Train Epoch: {} Loss: {:.6f}\t".format(
                    epoch,
                    train_loss / (batch_idx + 1),
                )
            )
    train_accuracy = correct / total
    train_loss = metric_average(train_loss)
    train_accuracy = metric_average(train_accuracy)
    # Bluefog: average metric values across workers.
    if bfl.rank() == 0:
        losses.append(train_loss / len(train_loader))
        accuracies.append(train_accuracy)
        print(
            "\nTrain Epoch: {}\tAverage loss: {:.6f}\tAccuracy: {:.4f}%\n".format(
                epoch, train_loss / len(train_loader), 100.0 * train_accuracy
            ),
            flush=True,
        )

def test(epoch):
    model.eval()
    test_loss, correct, total = 0, 0, 0
    for data, targets in test_loader:
        data, targets = data.cuda(), targets.cuda()
        outputs = model(data)
        loss = F.cross_entropy(outputs, targets)

        test_loss += loss.item()
        _, pred = outputs.max(dim=1)
        total += targets.size(dim=0)
        correct += pred.eq(targets).sum().item()

    test_accuracy = correct / total
    test_loss = metric_average(test_loss)
    test_accuracy = metric_average(test_accuracy)
    # Bluefog: average metric values across workers.
    if bfl.rank() == 0:
        print(
            "\nTest Epoch: {}\tAverage loss: {:.6f}\tAccuracy: {:.4f}%\n".format(
                epoch, test_loss / len(test_loader), 100.0 * test_accuracy
            ),
            flush=True,
        )

start = time.time()
for e in range(epochs):
    
    train(e)
    test(e)
end = time.time()
print("time of training is {}.epoch:".format(end-start,epochs))

    
bfl.barrier(device=device)
if bfl.rank()==0:
    f=open("/home/qyq/bfl/results/{}_loss.txt".format(args.experiment_name),'w')
    for loss in losses:
        f.write(str(loss)+'\n')
    f.close()
    f=open("/home/qyq/bfl/results/{}_accuracy.txt".format(args.experiment_name),'w')
    for accuracy in accuracies:
        f.write(str(accuracy)+'\n')
    f.close()
print(f"rank {bfl.rank()} finished.")
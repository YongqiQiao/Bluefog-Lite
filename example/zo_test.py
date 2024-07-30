
from torchvision import datasets, transforms
import torch.nn.functional as F
from torchvision.models import resnet18,resnet34,resnet50,resnet101,resnet152
import torch.utils.data
from torch import nn
import numpy as np
import time
from model import CNN_MNIST



# Load the dataset
my_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
train_data = datasets.MNIST('/home/qyq/bfl/data', train=True, download=True, transform=my_transform)
test_data = datasets.MNIST('/home/qyq/bfl/data', train=False, download=True, transform=my_transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)


# Load the model
model = CNN_MNIST()
# in_features = model.fc.in_features
# model.fc = nn.Linear(in_features, 10)
model.cuda()
named_parameters_to_optim = []
for name,param in model.named_parameters():
    if param.requires_grad:
        named_parameters_to_optim.append((name, param))
        param.grad=None
# hyper-parameters

lr = 0.001
epochs = 1000
shrink_factor = 0.85
num_perturb = 20
inc_rate = 0.4
zo_eps = 0.0001
weight_decay=1e-5
projected_grad_list = torch.zeros(num_perturb)
projected_grad_list.cuda(0)



optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
print("hyper-parameters seted")

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
        # First function evaluation
        start = time.time()
        zo_perturb_parameters(scaling_factor=1, random_seed=zo_random_seed)
        end = time.time()
        # print("time of zo_perturb_parameters is {}".format(end-start))
        start = time.time()
        loss1 = zo_forward(inputs, targets)
        end = time.time()
        # print("time of zo_forward is {}".format(end-start))

        # Second function evaluation
        zo_perturb_parameters(scaling_factor=-2, random_seed=zo_random_seed)
        loss2 = zo_forward(inputs, targets)

        projected_grad = ((loss1 - loss2) / (2 * zo_eps)).item()
        projected_grad_list[i] = projected_grad
        zo_perturb_parameters(scaling_factor=1, random_seed=zo_random_seed)
    return loss1

def zo_update(epoch):
    """
    Update the parameters with the estimated gradients.
    """
    from collections import defaultdict
    grad_tmp = defaultdict(float)

    # Reset the random seed for sampling zs
    for i in range(num_perturb):
        start = time.time()
        torch.manual_seed((epoch+1)*(i+1))
        for name, param in named_parameters_to_optim:
            # Resample z
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device,
                            dtype=param.data.dtype)
            grad_tmp[name] += projected_grad_list[i] * z
        end = time.time()
        # print("time of resample z is {}".format(end-start))
    start = time.time()
    for name, param in named_parameters_to_optim:
        if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
            param.data = param.data - lr * (
                grad_tmp[name] / num_perturb + weight_decay * param.data)
        else:
            param.data = param.data - lr * (grad_tmp[name] / num_perturb)
    end = time.time()
    # print("time of zo_update is {}".format(end-start))


def train(epoch):
    model.train()
    train_loss, correct, total = 0, 0, 0
    for batch_idx, (data,targets) in enumerate(train_loader):
        data, targets = data.cuda(), targets.cuda()
        loss = zo_step(data, targets, epoch)
        zo_update(epoch)
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
    # Bluefog: average metric values across workers.
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
    # Bluefog: average metric values across workers.
    print(
        "\nTest Epoch: {}\tAverage loss: {:.6f}\tAccuracy: {:.4f}%\n".format(
            epoch, test_loss / len(test_loader), 100.0 * test_accuracy
        ),
        flush=True,
    )


for e in range(epochs):
    print("start training")
    train(e)
    test(e)




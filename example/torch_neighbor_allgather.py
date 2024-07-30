import torch.distributed as dist
import torch
import bluefoglite
import bluefoglite.torch_api as bfl

bfl.init()
bfl.set_topology(topology=bluefoglite.RingGraph(bfl.size()))
topo = bfl.load_topology()
self_weight,send_weights = bluefoglite.GetSendWeights(topo, bfl.rank())
print("rank:{},send_weights:{}".format(bfl.rank(),send_weights))

neighbor_size = len(send_weights.keys())
print(f"I am rank {bfl.rank()} among size {bfl.size()}.")

x = torch.nn.Parameter(
    torch.arange(bfl.rank(), bfl.rank() + 4).float(), requires_grad=True
)

y = torch.dot(x, x)
y.backward()

print("Rank ", bfl.rank(), " x.data: ", x.data)
print("Rank ", bfl.rank(), " x.grad: ", x.grad)
data_buffer = [torch.zeros(len(x.data)) for _ in range(neighbor_size)]
grad_buffer = [torch.zeros(len(x.grad)) for _ in range(neighbor_size)]

bfl.neighbor_allgather(data_buffer, x.data, inplace=True)
bfl.neighbor_allgather(grad_buffer, x.grad, inplace=True)
print("Rank ", bfl.rank(), " data_buffer: ", data_buffer)
print("Rank ", bfl.rank(), " grad_buffer: ", grad_buffer)
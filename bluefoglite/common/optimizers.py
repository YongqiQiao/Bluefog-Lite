from enum import Enum
from collections import Counter
import itertools
import warnings
from contextlib import contextmanager

import torch
import bluefoglite.torch_api as bfl


class CommunicationType(Enum):
    neighbor_allreduce = "neighbor.allreduce"
    allreduce = "allreduce"
    empty = "empty"


_warning_message_num_step_per_communication = (
    "Unexpected behavior:\n"
    "  After num_steps_per_communication times of forward computation `y=model(x)` are called,\n"
    "  an optimizer step() function must be called.\n"
    "  It does not matter how many step() functions are called in between.\n"
    "  Please adjust num_step_per_communication to update model parameters locally.\n"
    "  More information can be found in the FAQ page.\n"
)
_warning_message_backward_pass_per_step = (
    "Unexpected behavior:\n"
    "  After num_steps_per_communication times of backward"
    " computation `loss.backward()` are called,\n"
    "  an optimizer step() function must be called.\n"
    "  It does not matter how many step() functions are called in between.\n"
    "  Please adjust num_steps_per_communication to accumulate gradients locally.\n"
    "  More information can be found in the FAQ page.\n"
)


def _named_leaf_module(module, parent_name=None):
    """Yield an iterator over all leaf modules."""
    if not list(module.named_children()):
        yield (parent_name, module)
    for name, ch_module in module.named_children():
        full_name = parent_name + "." + name if parent_name else name
        yield from _named_leaf_module(ch_module, full_name)


def _check_named_parameters(optimizer, model):
    _models = None
    if isinstance(model, torch.nn.Module):
        _models = [model]
    if isinstance(model, list):
        for m in model:
            assert isinstance(m, torch.nn.Module)
        _models = model
    assert _models is not None
    named_parameters = list(itertools.chain(*[m.named_parameters() for m in _models]))

    # make sure that named_parameters are tuples
    if any(not isinstance(p, tuple) for p in named_parameters):
        raise ValueError(
            "named_parameters should be a sequence of "
            "tuples (name, parameter), usually produced by "
            "model.named_parameters()."
        )

    name_list = [k for k, _ in named_parameters]
    name_list_remove_dups = list({k for k, _ in named_parameters})
    diff_counter = Counter(name_list) - Counter(name_list_remove_dups)
    dups = list(diff_counter.elements())
    if dups:
        raise ValueError(
            "Parameter names in named_parameters must be unique. "
            f"Found duplicates: {', '.join(dups)}"
        )

    all_param_ids = {
        id(v) for param_group in optimizer.param_groups for v in param_group["params"]
    }
    named_param_ids = {id(v) for k, v in named_parameters}
    unnamed_param_ids = all_param_ids - named_param_ids
    if unnamed_param_ids:
        raise ValueError(
            "Named parameters provided by model are mismatch with the parameters"
            "handled by optimizer. Python object ids: "
            f"{', '.join(str(id) for id in unnamed_param_ids)}"
        )
    return named_parameters, _models


# pylint: disable=too-many-instance-attributes
class _DistributedReduceOptimizer(torch.optim.Optimizer):
    def __init__(
        self, params, model, communication_type, num_steps_per_communication=1
    ):
        # pylint: disable=bad-super-call, no-value-for-parameter
        super(self.__class__, self).__init__(params)

        named_parameters, models = _check_named_parameters(self, model)
        # knobs for neighbor communication behavior
        self.self_weight = None
        self.src_weights = None
        self.dst_weights = None
        self.src_machine_weights = None
        self.dst_machine_weights = None
        self.enable_topo_check = False

        self._models = models
        self._parameter_names = {v: k for k, v in sorted(named_parameters)}
        self._name_parameters = dict(sorted(named_parameters))
        self._async_works = {}
        self._requires_update = set()
        self._synchronized = False
        self._should_synchronize = True
        self._error_encountered = False
        self._num_steps_per_communication = num_steps_per_communication
        assert isinstance(communication_type, CommunicationType)
        self._communication_type = communication_type

        if bfl.size() > 1:
            self._register_hooks()

    def _register_hooks(self):
        for model in self._models:
            # The hook is added at model level instead of layer level, as it avoids triggering
            # the hook function of the same layer multiple times in case the layer is called
            # several times during the forward computation of the model.
            model.register_forward_hook(self._make_hook())
            self._requires_update.update(dict(model.named_parameters()).values())

    def _make_hook(self):
        def hook(model, *unused):
            for parent_name, layer in _named_leaf_module(model):
                for name, p in layer.named_parameters():
                    if not layer.training:
                        continue
                    if (
                        self._name_parameters.get(parent_name + "." + name, None)
                        is None
                    ):
                        # Some case like encoder-decode, which shared the same weights.
                        continue
                    if p.requires_grad:
                        if self._communication_type == CommunicationType.allreduce:
                            async_work = self._allreduce_data_async(p)
                        elif (
                            self._communication_type
                            == CommunicationType.neighbor_allreduce
                        ):
                            async_work = self._neighbor_allreduce_data_async(p)
                        else:
                            raise ValueError(
                                "Unsuppported CommunicationType encountered."
                            )
                        self._async_works[p] = async_work

        return hook

    def _neighbor_allreduce_data_async(self, p):
        async_work = bfl.neighbor_allreduce_nonblocking(
            p.data,
            self_weight=self.self_weight,
            src_weights=self.src_weights,
            dst_weights=self.dst_weights,
            inplace=True,
        )
        return async_work

    def _allreduce_data_async(self, p):
        async_work = bfl.allreduce_nonblocking(p.data, inplace=True)
        return async_work

    @property
    def communication_type(self):
        return self._communication_type

    @communication_type.setter
    def communication_type(self, value):
        assert isinstance(value, CommunicationType)
        self._communication_type = value

    def synchronize(self):
        with torch.no_grad():
            for _, async_work in self._async_works.items():
                if async_work is not None:
                    async_work.wait()
        self._async_works.clear()
        self._synchronized = True

    @contextmanager
    def skip_synchronize(self):
        """
        A context manager used to specify that optimizer.step() should
        not perform synchronization.

        It's typically used in a following pattern:

        .. code-block:: python

            optimizer.synchronize()
            with optimizer.skip_synchronize():
                optimizer.step()
        """
        self._should_synchronize = False
        try:
            yield
        finally:
            self._should_synchronize = True

    def step(self, closure=None):
        # consensus style is the easiest way to implement it.
        if self._should_synchronize:
            if self._synchronized:
                warnings.warn(
                    "optimizer.step() called without "
                    "optimizer.skip_synchronize() context after "
                    "optimizer.synchronize(). This can cause training "
                    "slowdown. You may want to consider using "
                    "optimizer.skip_synchronize() context if you use "
                    "optimizer.synchronize() in your code."
                )
            self.synchronize()
        self._synchronized = False
        # pylint: disable=bad-super-call
        return super(self.__class__, self).step(closure)


# pylint: disable=too-many-instance-attributes
class _DistributedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, model, backward_passes_per_step=1):
        # pylint: disable=bad-super-call, no-value-for-parameter
        super(self.__class__, self).__init__(params)

        named_parameters, models = _check_named_parameters(self, model)
        self._models = models
        self._parameter_names = {v: k for k, v in sorted(named_parameters)}
        self._async_works = {}
        self._grad_accs = []
        self._requires_update = set()
        self._synchronized = False
        self._should_synchronize = True
        self._backward_passes_per_step = backward_passes_per_step
        self._error_encountered = False

        if bfl.size() > 1:
            self._register_hooks()

    def _register_hooks(self):
        for param_group in self.param_groups:
            for p in param_group["params"]:
                if p.requires_grad:
                    p.grad = p.data.new(p.size()).zero_()
                    self._requires_update.add(p)
                    p_tmp = p.expand_as(p)
                    grad_acc = p_tmp.grad_fn.next_functions[0][0]
                    grad_acc.register_hook(self._make_hook(p))
                    self._grad_accs.append(grad_acc)

    def _make_hook(self, p):
        def hook(*ignore):
            assert not p.grad.requires_grad
            async_work = self._allreduce_grad_async(p)
            self._async_works[p] = async_work

        return hook

    def _allreduce_grad_async(self, p):
        async_work = bfl.allreduce_nonblocking(p.grad, inplace=True)
        return async_work

    def synchronize(self):
        with torch.no_grad():
            for _, async_work in self._async_works.items():
                async_work.wait()
        self._async_works.clear()
        self._synchronized = True

    @contextmanager
    def skip_synchronize(self):
        """
        A context manager used to specify that optimizer.step() should
        not perform synchronization.

        It's typically used in a following pattern:

        .. code-block:: python

            optimizer.synchronize()
            with optimizer.skip_synchronize():
                optimizer.step()
        """
        self._should_synchronize = False
        try:
            yield
        finally:
            self._should_synchronize = True

    def step(self, closure=None):
        if self._should_synchronize:
            if self._synchronized:
                warnings.warn(
                    "optimizer.step() called without "
                    "optimizer.skip_synchronize() context after "
                    "optimizer.synchronize(). This can cause training "
                    "slowdown. You may want to consider using "
                    "optimizer.skip_synchronize() context if you use "
                    "optimizer.synchronize() in your code."
                )
            self.synchronize()
        self._synchronized = False
        # pylint: disable=bad-super-call
        return super(self.__class__, self).step(closure)


def DistributedAdaptWithCombineOptimizer(
    optimizer,
    model,
    communication_type=CommunicationType.neighbor_allreduce,
    num_steps_per_communication=1,
):
    cls = type(
        optimizer.__class__.__name__,
        (optimizer.__class__,),
        dict(_DistributedReduceOptimizer.__dict__),
    )
    return cls(
        optimizer.param_groups, model, communication_type, num_steps_per_communication
    )


def DistributedGradientAllreduceOptimizer(
    optimizer,
    model,
    num_steps_per_communication=1,
):
    cls = type(
        optimizer.__class__.__name__,
        (optimizer.__class__,),
        dict(_DistributedOptimizer.__dict__),
    )
    return cls(optimizer.param_groups, model, num_steps_per_communication)

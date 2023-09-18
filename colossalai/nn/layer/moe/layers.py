import math
from typing import Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from colossalai.context.moe_context import MOE_CONTEXT
from colossalai.legacy.zero.init_ctx import no_shard_zero_context, no_shard_zero_decrator
from colossalai.nn.layer.moe._operation import (
    COL_MOE_KERNEL_FLAG,
    AllGather,
    AllToAll,
    MoeCombine,
    MoeDispatch,
    ReduceScatter,
)
from colossalai.nn.layer.moe.experts import Experts, MoeExperts
from colossalai.nn.layer.moe.routers import MoeRouter, Top1Router, Top2Router
from colossalai.nn.layer.moe.utils import NormalNoiseGenerator, UniformNoiseGenerator
from colossalai.utils import get_current_device


@no_shard_zero_decrator(is_replicated=True)
class MoeLayer(nn.Module):
    """A MoE layer, that puts its input tensor to its gate and uses the output logits
    to router all tokens, is mainly used to exchange all tokens for every expert across
    the moe tensor group by all to all communication. Then it will get the output of all
    experts and exchange the output. At last returns the output of the moe system.

    Args:
        dim_model (int): Dimension of model.
        num_experts (int): The number of experts.
        router (MoeRouter): Instance of router used in routing.
        experts (MoeExperts): Instance of experts generated by Expert.
    """

    def __init__(self, dim_model: int, num_experts: int, router: MoeRouter, experts: MoeExperts):
        super().__init__()
        self.d_model = dim_model
        self.num_experts = num_experts
        self.gate_weight = torch.nn.Parameter(torch.empty(num_experts, dim_model))
        self.router: MoeRouter = router
        self.experts: MoeExperts = experts
        self.use_kernel = True if COL_MOE_KERNEL_FLAG and MOE_CONTEXT.use_kernel_optim else False
        self.ep_group = experts.dist_info.ep_group
        self.ep_size = experts.dist_info.ep_size
        self.num_local_experts = experts.num_local_experts

        nn.init.trunc_normal_(self.gate_weight, std=math.sqrt(0.1 / dim_model))

    def a2a_process(self, dispatch_data: torch.Tensor):
        expert_input = AllToAll.apply(dispatch_data, self.ep_group)
        input_shape = expert_input.shape
        expert_input = expert_input.reshape(self.ep_size, self.num_local_experts, -1, self.d_model)
        expert_output = self.experts(expert_input)
        expert_output = expert_output.reshape(input_shape)
        expert_output = AllToAll.apply(expert_output, self.ep_group)
        return expert_output

    def tp_process(self, dispatch_data: torch.Tensor):
        expert_in = AllGather.apply(dispatch_data, self.ep_group)
        expert_out = self.experts(expert_in)
        expert_out = ReduceScatter.apply(expert_out, self.ep_group)
        return expert_out

    def forward(self, inputs: torch.Tensor) -> Tuple:
        # reshape the input tokens
        tokens = inputs.reshape(-1, self.d_model)

        # the data type of the inputs in the gating should be fp32
        fp32_input = tokens.to(torch.float)
        fp32_weight = self.gate_weight.to(torch.float)
        gate_output = F.linear(fp32_input, fp32_weight)

        # the result from the router
        route_result_list = self.router(inputs=gate_output, use_kernel=self.use_kernel, ep_group=self.ep_group)

        if self.use_kernel:
            dispatch_data = MoeDispatch.apply(tokens, *route_result_list[1:])
            dispatch_data = dispatch_data.reshape(self.num_experts, -1, self.d_model)
        else:
            sec_mask_f = route_result_list[1].type_as(inputs)
            dispatch_data = torch.matmul(sec_mask_f.permute(1, 2, 0), tokens)

        # dispatch_data [e, c, h]
        if self.experts.comm_name == "all_to_all":
            expert_output = self.a2a_process(dispatch_data)
        elif self.experts.comm_name == "all_gather":
            expert_output = self.tp_process(dispatch_data)
        else:
            raise NotImplementedError("This kind of communication has not been implemented yet.\n Please use Experts "
                                      "build function.")
        # expert_output [e, c, h]
        if self.use_kernel:
            expert_output = expert_output.reshape(-1, self.d_model)
            ans = MoeCombine.apply(expert_output, *route_result_list)
        else:
            combine_weights = route_result_list[0].type_as(inputs)
            combine_weights = combine_weights.view(combine_weights.shape[0], -1)
            expert_output = expert_output.view(-1, expert_output.shape[-1])
            ans = torch.matmul(combine_weights, expert_output)

        ans = ans.reshape(inputs.shape)
        l_aux = self.router.pop_routing_loss()
        return ans, l_aux


class MoeModule(nn.Module):
    """A class for users to create MoE modules in their models.

    Args:
        dim_model (int): Hidden dimension of training model
        num_experts (int): The number experts
        top_k (int, optional): The number of experts for dispatchment of each token
        capacity_factor_train (float, optional): Capacity factor in routing during training
        capacity_factor_eval (float, optional): Capacity factor in routing during evaluation
        min_capacity (int, optional): The minimum number of the capacity of each expert
        noisy_policy (str, optional): The policy of noisy function. Now we have 'Jitter' and 'Gaussian'.
            'Jitter' can be found in `Switch Transformer paper`_.
            'Gaussian' can be found in `ViT-MoE paper`_.
        drop_tks (bool, optional): Whether drops tokens in evaluation
        use_residual (bool, optional): Makes this MoE layer a Residual MoE.
            More information can be found in `Microsoft paper`_.
        residual_instance (nn.Module, optional): The instance of residual module in Residual MoE
        expert_instance (MoeExperts, optional): The instance of experts module in MoeLayer
        expert_cls (Type[nn.Module], optional): The class of each expert when no instance is given
        expert_args (optional): The args of expert when no instance is given

    .. _Switch Transformer paper:
        https://arxiv.org/abs/2101.03961
    .. _ViT-MoE paper:
        https://arxiv.org/abs/2106.05974
    .. _Microsoft paper:
        https://arxiv.org/abs/2201.05596
    """

    def __init__(self,
                 dim_model: int,
                 num_experts: int,
                 top_k: int = 1,
                 capacity_factor_train: float = 1.25,
                 capacity_factor_eval: float = 2.0,
                 min_capacity: int = 4,
                 noisy_policy: Optional[str] = None,
                 drop_tks: bool = True,
                 use_residual: bool = False,
                 residual_instance: Optional[nn.Module] = None,
                 expert_instance: Optional[MoeExperts] = None,
                 expert_cls: Optional[Type[nn.Module]] = None,
                 **expert_args):
        super().__init__()

        noisy_func = None
        if noisy_policy is not None:
            if noisy_policy == 'Jitter':
                noisy_func = UniformNoiseGenerator()
            elif noisy_policy == 'Gaussian':
                noisy_func = NormalNoiseGenerator(num_experts)
            else:
                raise NotImplementedError("Unsupported input noisy policy")

        if top_k == 1:
            moe_router_cls = Top1Router
        elif top_k == 2:
            moe_router_cls = Top2Router
        else:
            raise NotImplementedError("top_k > 2 is not supported yet")

        self.moe_router = moe_router_cls(capacity_factor_train=capacity_factor_train,
                                         capacity_factor_eval=capacity_factor_eval,
                                         min_capacity=min_capacity,
                                         noisy_func=noisy_func,
                                         drop_tks=drop_tks)
        self.use_residual = use_residual
        if use_residual:
            if residual_instance is not None:
                self.residual_module = residual_instance
            else:
                assert expert_cls is not None, \
                    "Expert class can't be None when residual instance is not given"
                self.residual_module = expert_cls(**expert_args)

            with no_shard_zero_context():
                self.residual_combine = nn.Linear(dim_model, 2, device=get_current_device())

        if expert_instance is not None:
            my_experts = expert_instance
        else:
            assert expert_cls is not None, \
                "Expert class can't be None when experts instance is not given"
            my_experts = Experts(expert_cls, num_experts, **expert_args)

        self.moe_layer = MoeLayer(dim_model=dim_model,
                                  num_experts=num_experts,
                                  router=self.moe_router,
                                  experts=my_experts)

    def forward(self, inputs: torch.Tensor):
        moe_output, l_aux = self.moe_layer(inputs)

        if self.use_residual:
            residual_output = self.residual_module(inputs)
            combine_coef = self.residual_combine(inputs)
            combine_coef = F.softmax(combine_coef, dim=-1)
            output = moe_output * combine_coef[..., 0:1] + residual_output * combine_coef[..., 1:]
        else:
            output = moe_output

        return output, l_aux

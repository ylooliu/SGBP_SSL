import torch
from utils import global_v as glv
import numpy as np

class SGBP_SSL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, network_config, layer_config,sg_p):
        shape = inputs.shape
        n_steps = shape[4] 
        theta_m = 1/network_config['tau_m']
        tau_s = network_config['tau_s']
        theta_s = 1/tau_s
        threshold = layer_config['threshold']
        vrest=torch.tensor(0)

        mem = (vrest*torch.ones(shape[0], shape[1], shape[2], shape[3])).cuda()
        syn = torch.zeros(shape[0], shape[1], shape[2], shape[3]).cuda()
        syns_posts = []
        mems = []
        mem_updates = []
        outputs = []
        for t in range(n_steps):
            mem_update = (-theta_m) * mem + inputs[..., t]
            mem += mem_update

            out = mem > threshold
            out = out.type(torch.float32)

            mems.append(mem)

            mem = mem * (1-out)+vrest*out
            outputs.append(out)

            mem_updates.append(mem_update)
            syn = syn + (out - syn) * theta_s
            syns_posts.append(syn)

        mems = torch.stack(mems, dim = 4)
        mem_updates = torch.stack(mem_updates, dim = 4)
        outputs = torch.stack(outputs, dim = 4)
        syns_posts = torch.stack(syns_posts, dim = 4)
        if sg_p is not None:
            ctx.save_for_backward(mem_updates, outputs, mems, syns_posts, torch.tensor([threshold, tau_s, theta_m,sg_p]))
        else:
            ctx.save_for_backward(mem_updates, outputs, mems, syns_posts,torch.tensor([threshold, tau_s, theta_m]))
        return syns_posts

    @staticmethod
    def backward(ctx, grad_delta):  # grad_delta=L/a =L/u_j *u_j/a
        """ 实现反向传播， grad_delta为损失对前向传播中输出syns_posts的梯度，但需要计算的是损失对输入inputs的梯度 """

        (delta_u, outputs, u, syns, others) = ctx.saved_tensors
        shape = grad_delta.shape

        n_steps = shape[4]

        threshold = others[0].item()
        tau_s = others[1].item()
        theta_m = others[2].item()
        sg_p=others[3].item()
        vrest = torch.tensor(0)

        th = 1 / (4 * tau_s)

        grad = torch.zeros_like(grad_delta)

        syn_a = glv.syn_a.repeat(shape[0], shape[1], shape[2], shape[3], 1)
        partial_a = glv.syn_a / (-tau_s)
        partial_a = partial_a.repeat(shape[0], shape[1], shape[2], shape[3], 1) * outputs

        theta = torch.zeros(shape[0], shape[1], shape[2], shape[3]).cuda()
        for t in range(n_steps - 1, -1, -1):
            list1 = []
            list2=[]
            time_end = n_steps
            time_len = time_end - t

            out = outputs[..., t]
            partial_u = torch.clamp(-1 / delta_u, -8, 0)
            partial_a_partial_u = partial_a * partial_u

            for t_p in range(t, n_steps, 1):
                if t_p == t:
                    partial_uhtk_partial_uhtm = torch.ones_like(u[..., 0]).cuda()
                elif t_p == t + 1:
                    sg = (1 / sg_p) * torch.sign(torch.abs(u[..., t] - threshold) < (sg_p/ 2))
                    partial_uhtk_partial_uhtm = (1 - theta_m) * (1 - out) + (vrest - u[..., t + 1]) * sg
                else:
                    f1 = (1  - theta_m)*(1-outputs[..., t :t_p-1])
                    sg = (1 /sg_p) * torch.sign(torch.abs(u[..., t:t_p - 1] - threshold) < (sg_p / 2))
                    partial_uhtk_partial_uhtm = torch.prod(f1 + (vrest - u[..., t + 1:t_p]) * sg, dim=4)

                partial_atk_partial_utk=partial_a_partial_u[..., t_p]*tau_s

                list1.append(partial_atk_partial_utk * partial_uhtk_partial_uhtm)
                list2.append(torch.sum(grad_delta[..., t_p:time_end],dim=-1))

            partial_a_partial_u = torch.stack(list1, dim=4)
            grad_delta_t=torch.stack(list2,dim=4)

            grad_tmp = torch.sum(partial_a_partial_u*grad_delta_t, dim=4)      #就是文章里的fai

            grad_a = torch.sum(syn_a[..., 0:time_len] * grad_delta[..., t:time_end], dim=-1)

            a = 0.2
            f = torch.clamp((-1 * u[..., t] + threshold) / a, -8, 8)
            f = torch.exp(f)
            f = f / ((1 + f) * (1 + f) * a)

            grad_a = grad_a * f
            syn = syns[..., t]

            grad_tmp[syn < th] = grad_a[syn < th]
            grad[..., t] = grad_tmp


        return grad, None, None,None



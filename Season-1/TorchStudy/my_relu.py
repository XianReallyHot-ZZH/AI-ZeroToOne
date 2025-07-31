# custom ReLU function with autograd
import torch

class MyReLU(torch.autograd.Function):          # 继承 torch.autograd.Function 类
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)                # 将输入张量保存到上下文对象中，以便在反向传播时使用
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors                      # 从上下文中取出之前保存的输入张量
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input
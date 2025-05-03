import numpy as np
import torch


class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W  # torch.Tensor
        self.b = b  # torch.Tensor
        self.stride = stride
        self.pad = pad

        # 中间数据
        self.x = None
        self.col = None
        self.col_W = None

        # 梯度
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + (H + 2 * self.pad - FH) // self.stride
        out_w = 1 + (W + 2 * self.pad - FW) // self.stride

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.view(FN, -1).T

        out = torch.matmul(col, col_W) + self.b
        out = out.view(N, out_h, out_w, -1).permute(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.permute(0, 2, 3, 1).reshape(-1, FN)

        self.db = torch.sum(dout, dim=0)
        self.dW = torch.matmul(self.col.T, dout)
        self.dW = self.dW.permute(1, 0).view(FN, C, FH, FW)

        dcol = torch.matmul(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx

class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        self.out = 1 / (1 + torch.exp(-x))
        return self.out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.clone()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        return dout


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = torch.zeros_like(val)
                self.v[key] = torch.zeros_like(val)

        self.iter += 1
        lr_t = self.lr * torch.sqrt(torch.tensor(1.0 - self.beta2 ** self.iter)) / (1.0 - self.beta1 ** self.iter)

        for key in params.keys():
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)

            params[key] -= lr_t * self.m[key] / (torch.sqrt(self.v[key]) + 1e-7)


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """使用PyTorch实现im2col"""
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h) // stride + 1
    out_w = (W + 2*pad - filter_w) // stride + 1

    # 使用PyTorch的pad函数
    img = torch.nn.functional.pad(input_data, (pad, pad, pad, pad), mode='constant', value=0)
    col = torch.zeros((N, C, filter_h, filter_w, out_h, out_w), device=input_data.device)

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.permute(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col

def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """使用PyTorch实现col2im"""
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h) // stride + 1
    out_w = (W + 2*pad - filter_w) // stride + 1
    col = col.view(N, out_h, out_w, C, filter_h, filter_w).permute(0, 3, 4, 5, 1, 2)

    img = torch.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1), device=col.device)
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]


class BatchNormalization:
    def __init__(self, num_features, momentum=0.9, device=None):
        # 初始化可学习参数
        self.gamma = torch.ones(num_features, device=device)  # 缩放参数
        self.beta = torch.zeros(num_features, device=device)  # 平移参数
        self.momentum = momentum  # 动量参数
        self.eps = 1e-5  # 防止除零

        # 运行时统计量
        self.running_mean = torch.zeros(num_features, device=device)
        self.running_var = torch.ones(num_features, device=device)

        # 反向传播中间变量
        self.x_hat = None
        self.x_centered = None
        self.std = None
        self.input_shape = None

        # 梯度
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=False):
        self.input_shape = x.shape

        # 处理4D输入 (N, C, H, W) -> 转换为2D (N*H*W, C)
        if x.dim() == 4:
            N, C, H, W = x.shape
            x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)
        else:
            x_flat = x.reshape(-1, x.shape[-1])

        if train_flg:
            # 训练模式：计算当前batch统计量
            mean = x_flat.mean(dim=0)
            var = x_flat.var(dim=0, unbiased=False)

            # 更新运行时统计量
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            # 测试模式：使用运行时统计量
            mean = self.running_mean
            var = self.running_var

        # 归一化计算
        self.x_centered = x_flat - mean
        self.std = torch.sqrt(var + self.eps)
        self.x_hat = self.x_centered / self.std

        # 缩放和平移
        out = self.gamma * self.x_hat + self.beta

        # 恢复原始形状
        if x.dim() == 4:
            out = out.reshape(N, H, W, C).permute(0, 3, 1, 2)

        return out

    def backward(self, dout):
        # 处理4D梯度输入
        if dout.dim() == 4:
            N, C, H, W = dout.shape
            dout_flat = dout.permute(0, 2, 3, 1).reshape(-1, C)
        else:
            dout_flat = dout.reshape(-1, dout.shape[-1])

        # 计算梯度
        self.dbeta = torch.sum(dout_flat, dim=0)
        self.dgamma = torch.sum(dout_flat * self.x_hat, dim=0)

        dx_hat = dout_flat * self.gamma
        dx_centered = dx_hat / self.std

        dstd = -torch.sum(dx_hat * self.x_centered / (self.std ** 2), dim=0)
        dvar = 0.5 * dstd / self.std

        dx_centered += (2.0 / dout_flat.shape[0]) * self.x_centered * dvar

        dmean = torch.sum(dx_centered, dim=0)
        dx = dx_centered - dmean / dout_flat.shape[0]

        # 恢复原始形状
        if len(self.input_shape) == 4:  # 修改此处
            dx = dx.reshape(self.input_shape)


        return dx
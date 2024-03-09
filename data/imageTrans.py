import torch

def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (torch.arange(kernel_size).reshape(-1, 1),
          torch.arange(kernel_size).reshape(1, -1))
    filt = (1 - torch.abs(og[0] - center) / factor) * \
           (1 - torch.abs(og[1] - center) / factor)
    weight = torch.zeros(
        (in_channels, out_channels, kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return weight

def bilinear_upsampling(in_channels, out_channels, kernel_size=4, padding=1, stride=2, bias=False):
    """upsampling"""
    conv_trans = torch.nn.ConvTranspose2d(3, 3, kernel_size=kernel_size, padding=padding,
                                          stride=stride, bias=bias)
    conv_trans.weight.data.copy_(bilinear_kernel(in_channels=in_channels, out_channels=out_channels,
                                                 kernel_size=kernel_size))

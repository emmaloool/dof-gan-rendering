from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils
import math

# ------ from 16-726 HW3 ------
def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, norm='batch', init_zero_weights=False, spectral=False):
    """Creates a convolutional layer, with optional batch normalization.
    """
    layers = []
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    if init_zero_weights:
        conv_layer.weight.data = torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.001
    
    if spectral:
        conv_layer = utils.spectral_norm(conv_layer)

    layers.append(conv_layer)
    if norm == 'batch':
        layers.append(nn.BatchNorm2d(out_channels))
    elif norm == 'instance':
        layers.append(nn.InstanceNorm2d(out_channels))
    return nn.Sequential(*layers)

def up_conv(in_channels, out_channels, kernel_size, stride=1, padding=1, scale_factor=2, norm='batch'):
    """Creates a transposed-convolutional layer, with optional batch normalization.
    """
    layers = []
    blah = nn.Upsample(scale_factor=scale_factor, mode='nearest')
    layers.append(nn.Upsample(scale_factor=scale_factor, mode='nearest'))
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))
    if norm == 'batch':
        layers.append(nn.BatchNorm2d(out_channels))
    elif norm == 'instance':
        layers.append(nn.InstanceNorm2d(out_channels))

    return nn.Sequential(*layers)

# ------ Deconvolution is approximately same as the transpose of convolution? ------
def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1, norm='batch', init_zero_weights=False):
    """Creates a deconvolutional layer, with optional batch normalization.
    """
    layers = []
    conv_layer = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    if init_zero_weights:
        conv_layer.weight.data = torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.001
    layers.append(conv_layer)

    if norm == 'batch':
        layers.append(nn.BatchNorm2d(out_channels))
    elif norm == 'instance':
        layers.append(nn.InstanceNorm2d(out_channels))
    # elif norm == 'adain':
    #     layers.append(AdaIN(out_channels, 128))
    return nn.Sequential(*layers)


# # AdaIN implementation from https://github.com/CellEight/Pytorch-Adaptive-Instance-Normalization
# class AdaIN(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def mu(self, x):
#         """ Takes a (n,c,h,w) tensor as input and returns the average across
#         it's spatial dimensions as (h,w) tensor [See eq. 5 of paper]"""
#         return torch.sum(x,(2,3))/(x.shape[2]*x.shape[3])

#     def sigma(self, x):
#         """ Takes a (n,c,h,w) tensor as input and returns the standard deviation
#         across it's spatial dimensions as (h,w) tensor [See eq. 6 of paper] Note
#         the permutations are required for broadcasting"""
#         return torch.sqrt((torch.sum((x.permute([2,3,0,1])-self.mu(x)).permute([2,3,0,1])**2,(2,3))+0.000000023)/(x.shape[2]*x.shape[3]))

#     def forward(self, x, y):
#         """ Takes a content embeding x and a style embeding y and changes
#         transforms the mean and standard deviation of the content embedding to
#         that of the style. [See eq. 8 of paper] Note the permutations are
#         required for broadcasting"""
#         return (self.sigma(y)*((x.permute([2,3,0,1])-self.mu(x))/self.sigma(x)) + self.mu(y)).permute([2,3,0,1])

# EqualLR/EqualLinear + AdaIn implementations from https://github.com/rosinality/style-based-gan-pytorch/blob/master/model.py#:~:text=class%20AdaptiveInstanceNorm%28nn,return%20out
class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)

def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module

class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, input):
        # print("input: ", input.shape)
        return self.linear(input)

class AdaIN(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()

        self.norm = nn.InstanceNorm2d(in_channel)
        self.style = EqualLinear(style_dim, in_channel * 2)

        self.style.linear.bias.data[:in_channel] = 1
        self.style.linear.bias.data[in_channel:] = 0

    def forward(self, input, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = gamma * out + beta

        return out


# MLP implementation from https://colab.research.google.com/github/bentrevett/pytorch-image-classification/blob/master/1_mlp.ipynb#scrollTo=lAqzcW9XREvu
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.input_fc = nn.Linear(input_dim, 250)
        self.hidden_fc = nn.Linear(250, 100)
        self.output_fc = nn.Linear(100, output_dim)

    def forward(self, x):
        # x = [batch size, height, width]
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)

        # x = [batch size, height * width]
        h_1 = F.relu(self.input_fc(x))

        # h_1 = [batch size, 250]
        h_2 = F.relu(self.hidden_fc(h_1))

        # h_2 = [batch size, 100]
        y_pred = self.output_fc(h_2)

        # y_pred = [batch size, output dim]
        return y_pred

# module for center prior loss
class CenterFocusPriorLoss(nn.Module):
    def __init__(self, g, th):
        super().__init__()
        self.g = g
        self.th = th
       
    def forward(self, img):
        # calculate center of image
        device = img.device
        dim = img.shape[-1]
        center = dim / 2.

        # fast algorithm
        # this implements the following:
        # for x in range(dim):
        #     for y in range(dim):
                # r = math.sqrt((x - center)**2 + (y - center)**2)
                # if r <= self.th:
                #     center_prior[:, : , x, y] = 0.
                # else:
                #     center_prior[:, : , x, y] = -self.g * (r - self.th)
        
        # calculate grid of radii
        range = torch.arange(0, dim, device=device)
        x, y = torch.meshgrid(range, range)
        r = (x - center) ** 2 + (y - center) ** 2
        r = r.sqrt()
        # calculate center prior
        center_prior = -self.g * (r - self.th)
        center_prior[r <= self.th] = 0

        # replicate prior for all images in batch
        center_prior = center_prior.unsqueeze(0).repeat(img.shape[0], 1, 1, 1)

        # calculate loss
        return nn.MSELoss()(img, center_prior)

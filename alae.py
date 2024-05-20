
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from torch.nn.parameter import Parameter

import numpy as np

import math

import lreq as ln

from registry import *
# The paper consists of a few parts that require reading.

# First is ALAE itself. The architecture consists of a Generator and Discriminator modules.
# The Generator is split into two parts. A F mapping model that maps a latent space variable z
# into an intermediate latent space w, and a Generator model that takes the intermediate latent variable and some noise to generate an image.
# The Discriminator is split into two parts. An Encoder model that takes image input from the Generator and maps it into the w space.
# and a Discriminator model that discriminates the intermediate latent code from either a generated image or a real image.

# Next is StyleGan. The authors use Adaptive Instance Normalization to apply style injection into the models.
# This is optional since we do not know what styles we are aiming for in covid scans

# There is also techniques from Progressive Gan that help with steady training of GANs.
# 1. Progressive Growing Gans training paradigm by slowly building from low res images to high res ones.
# 2. Increasing variation using minibatch std dev
# 3. Normalization in generator and discriminator. Use equalized learning rate and pixelwise normalization in generator.

# Truncation Trick is used to help improve FID scores by limiting the range of values that are sampled for z.
# Z sampled during training is normal while during inference is truncated gaussian. Truncation psi are used during validation only

# To train ALAE in Progressive Gan paradigm, only Generator and Encoder are required to train by introducing lod's
# LOD counts start from 0.


def minibatch_stddev_layer(x, group_size=4):
    """
    Splits batches into <group_size> minibatches , and computes their standard deviation across group.

    Args:
        x (_type_): input tensor of shape [batch_size, channels, height, width].
        group_size (int, optional): size of splitted group. Defaults to 4.

    Returns:
        torch.Tensor: Original feature maps concatenated with additional <group_size> feature maps across dim=1, 
        shape=[batch_size, channels+group_size, height, width].
    """
    group_size = min(group_size, x.shape[0])
    size = x.shape[0]
    if x.shape[0] % group_size != 0:
        x = torch.cat(
            [x, x[:(group_size - (x.shape[0] % group_size)) % group_size]])
    y = x.view(group_size, -1, x.shape[1], x.shape[2], x.shape[3])
    y = y - y.mean(dim=0, keepdim=True)
    y = torch.sqrt((y ** 2).mean(dim=0) +
                   1e-8).mean(dim=[1, 2, 3], keepdim=True)
    y = y.repeat(group_size, 1, x.shape[2], x.shape[3])
    return torch.cat([x, y], dim=1)[:size]


def pixel_norm(x, epsilon=1e-8):
    """Does normalization for each feature map."""
    return x * torch.rsqrt(torch.mean(x.pow(2.0), dim=1, keepdim=True) + epsilon)


def style_mod(x, style):
    """Add styles to x

    Args:
        x (torch.Tensor): inputs
        style (torch.Tensor): style

    Returns:
        _type_: _description_
    """
    style = style.view(style.shape[0], 2, x.shape[1], 1, 1)
    return torch.addcmul(style[:, 1], value=1.0, tensor1=x, tensor2=style[:, 0] + 1)


def upscale2d(x, factor=2):
    # try both this and Upsample2d of pytorch as well
    s = x.shape
    x = torch.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
    x = x.repeat(1, 1, 1, factor, 1, factor)
    x = torch.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor])
    return x


def downscale2d(x, factor=2):
    return F.avg_pool2d(x, factor, factor)


class FromRGB(nn.Module):
    def __init__(self, channels, outputs):
        super(FromRGB, self).__init__()
        self.from_rgb = ln.Conv2d(channels, outputs, 1, 1, 0)

    def forward(self, x):
        x = self.from_rgb(x)
        x = F.leaky_relu(x, 0.2)

        return x


class ToRGB(nn.Module):
    def __init__(self, inputs, channels):
        super(ToRGB, self).__init__()
        self.inputs = inputs
        self.channels = channels
        self.to_rgb = ln.Conv2d(inputs, channels, 1, 1, 0, gain=0.03)

    def forward(self, x):
        x = self.to_rgb(x)
        return x


class Blur(nn.Module):
    # 3x3 2d blurring. register_buffer the weights makes the weights aka the kernel non-trainable
    def __init__(self, channels=3):
        """3x3 2D blurring
        Args:
            channels (int): number of channels. Defaults to 3.
        """
        super(Blur, self).__init__()
        f = np.array([1, 2, 1], dtype=np.float32)
        f = f[:, np.newaxis] * f[np.newaxis, :]
        f /= np.sum(f)
        kernel = torch.Tensor(f).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        self.register_buffer('weight', kernel)
        self.groups = channels

    def forward(self, x):
        return F.conv2d(x, weight=self.weight, groups=self.groups, padding=1)


# startf = starting resolution
# maxf = max resolution
# layer_count = number of layers in the encoder to grow.
# dlatent_size = latent w size (Confirm this later)
# fused_scale = use fused upscale and conv2d, else use separate ones




class DecodeBlock(nn.Module):
    def __init__(self, inputs, outputs, latent_size, has_first_conv=True, fused_scale=True, layer=0):
        """ Building blocks for Generator/Decoder. For every block with fused_scale and has_first_conv equals True, height and width is doubled
            
        Args:
            inputs (int): input channels
            outputs (int): output channels
            latent_size (int): latent size of w space
            has_first_conv (bool, optional): True if we want to include a conv layer before the final conv layer. Defaults to True.
            fused_scale (bool, optional): _description_. Defaults to True.
            layer (int, optional): _description_. Defaults to 0.
        """
        super(DecodeBlock, self).__init__()
        self.has_first_conv = has_first_conv
        self.inputs = inputs
        # use fused upscale and conv. Faster and less memory usage
        self.fused_scale = fused_scale
        if has_first_conv:
            if fused_scale:
                # upscale conv
                self.conv_1 = ln.ConvTranspose2d(
                    inputs, outputs, 3, 2, 1, bias=False, transform_kernel=True)
            else:
                # same size conv
                self.conv_1 = ln.Conv2d(inputs, outputs, 3, 1, 1, bias=False)

        self.blur = Blur(outputs)
        self.noise_weight_1 = nn.Parameter(torch.Tensor(1, outputs, 1, 1))
        self.noise_weight_1.data.zero_()
        self.bias_1 = nn.Parameter(torch.Tensor(1, outputs, 1, 1))
        self.instance_norm_1 = nn.InstanceNorm2d(
            outputs, affine=False, eps=1e-8)
        self.style_1 = ln.Linear(latent_size, 2 * outputs, gain=1)
        # same size conv
        self.conv_2 = ln.Conv2d(outputs, outputs, 3, 1, 1, bias=False)
        self.noise_weight_2 = nn.Parameter(torch.Tensor(1, outputs, 1, 1))
        self.noise_weight_2.data.zero_()
        self.bias_2 = nn.Parameter(torch.Tensor(1, outputs, 1, 1))
        self.instance_norm_2 = nn.InstanceNorm2d(
            outputs, affine=False, eps=1e-8)
        self.style_2 = ln.Linear(latent_size, 2 * outputs, gain=1)

        self.layer = layer

        with torch.no_grad():
            self.bias_1.zero_()
            self.bias_2.zero_()

    def forward(self, x, s1, s2, noise):
        """ Computes forward propagation of the decode block

        Args:
            x (torch.Tensor): inputs from last block.
            s1 (torch.Tensor): w space styles
            s2 (torch.Tensor): w space styles
            noise (torch.Tensor): Add noise to inputs between convs if True.

        Returns:
            torch.Tensor: outputs with shape=[batch_size, outputs, input_height*2, input_width*2]
        """
        if self.has_first_conv:
            if not self.fused_scale:
                x = upscale2d(x)
            x = self.conv_1(x)
            x = self.blur(x)

        # add noise to inputs
        if noise:
            if noise == 'batch_constant':
                x = torch.addcmul(x, value=1.0, tensor1=self.noise_weight_1,
                                  tensor2=torch.randn([1, 1, x.shape[2], x.shape[3]]))
            else:
                x = torch.addcmul(x, value=1.0, tensor1=self.noise_weight_1,
                                  tensor2=torch.randn([x.shape[0], 1, x.shape[2], x.shape[3]]))
        else:
            # normalizing
            s = math.pow(self.layer + 1, 0.5)
            x = x + s * torch.exp(-x * x / (2.0 * s * s)) / \
                math.sqrt(2 * math.pi) * 0.8
        x = x + self.bias_1

        x = F.leaky_relu(x, 0.2)

        x = self.instance_norm_1(x)

        # add styles to x
        x = style_mod(x, self.style_1(s1))

        x = self.conv_2(x)

        # add noise to inputs
        if noise:
            if noise == 'batch_constant':
                x = torch.addcmul(x, value=1.0, tensor1=self.noise_weight_2,
                                  tensor2=torch.randn([1, 1, x.shape[2], x.shape[3]]))
            else:
                x = torch.addcmul(x, value=1.0, tensor1=self.noise_weight_2,
                                  tensor2=torch.randn([x.shape[0], 1, x.shape[2], x.shape[3]]))
        else:
            s = math.pow(self.layer + 1, 0.5)
            x = x + s * torch.exp(-x * x / (2.0 * s * s)) / \
                math.sqrt(2 * math.pi) * 0.8

        x = x + self.bias_2

        x = F.leaky_relu(x, 0.2)
        x = self.instance_norm_2(x)

        x = style_mod(x, self.style_2(s2))

        return x

class EncodeBlock(nn.Module):
    def __init__(self, inputs, outputs, latent_size, last=False, fused_scale=True) -> None:
        super(EncodeBlock, self).__init__()
        # filter size = 3, stride = 1, padding = 1 is same size filter
        # filter size = 3, stride = 2, padding = 1 is same size half size filter
        self.conv_1 = ln.Conv2d(inputs, inputs, 3, 1, 1, bias=False)
        self.bias_1 = nn.Parameter(torch.Tensor(1, inputs, 1, 1))
        self.instance_norm_1 = nn.InstanceNorm2d(inputs, affine=False)
        self.blur = Blur(inputs)
        self.last = last
        self.fused_scale = fused_scale
        if last:
            self.dense = ln.Linear(inputs * 4 * 4, outputs)
        else:
            if fused_scale:
                self.conv_2 = ln.Conv2d(
                    inputs, outputs, 3, 2, 1, bias=False, transform_kernel=True)
            else:
                self.conv_2 = ln.Conv2d(inputs, outputs, 3, 1, 1, bias=False)

        self.bias_2 = nn.Parameter(torch.Tensor(1, outputs, 1, 1))
        self.instance_norm_2 = nn.InstanceNorm2d(outputs, affine=False)
        # 2* inputs is because of style shape is [batch_size, 2*inputs, 1, 1]
        self.style_1 = ln.Linear(2 * inputs, latent_size)
        if last:
            self.style_2 = ln.Linear(outputs, latent_size)
        else:
            self.style_2 = ln.Linear(2 * outputs, latent_size)

        with torch.no_grad():
            self.bias_1.zero_()
            self.bias_2.zero_()

    def forward(self, x):
        x = self.conv_1(x) + self.bias_1
        x = F.leaky_relu(x, 0.2)

        # calculate mean and stc deviation of feature maps
        # m.shape = [batch_size, channels, 1, 1]
        # std.shape = [batch_size, channels, 1, 1]
        # style_1.shape, style_2.shape = [batch_size, channels*2, 1, 1]
        # w space is mapped onto using style means and std deviations and passing that into linear layer
        m = torch.mean(x, dim=[2, 3], keepdim=True)
        std = torch.sqrt(torch.mean((x - m) ** 2, dim=[2, 3], keepdim=True))
        style_1 = torch.cat((m, std), dim=1)

        x = self.instance_norm_1(x)

        if self.last:
            x = self.dense(x.view(x.shape[0], -1))

            x = F.leaky_relu(x, 0.2)

            w1 = self.style_1(style_1.view(style_1.shape[0], style_1.shape[1]))
            w2 = self.style_2(x.view(x.shape[0], x.shape[1]))
        else:
            x = self.conv_2(self.blur(x))
            if not self.fused_scale:
                x = downscale2d(x)
            x = x + self.bias_2

            x = F.leaky_relu(x, 0.2)

            m = torch.mean(x, dim=[2, 3], keepdim=True)
            std = torch.sqrt(torch.mean(
                (x - m) ** 2, dim=[2, 3], keepdim=True))
            style_2 = torch.cat((m, std), dim=1)

            x = self.instance_norm_2(x)

            w1 = self.style_1(style_1.view(style_1.shape[0], style_1.shape[1]))
            w2 = self.style_2(style_2.view(style_2.shape[0], style_2.shape[1]))
            
        return x, w1, w2


@ENCODERS.register("EncoderNoStyle")
class EncoderNoStyle(nn.Module):
    def __init__(self, startf=32, maxf=256, layer_count=3, latent_size=512, channels=3):
        super(EncoderNoStyle, self).__init__()
        # Encoder encodes image to latent
        # if lod is low, the lower amount of encode blocks is needed because it is nearer to the output.
        self.maxf = maxf
        self.startf = startf
        self.layer_count = layer_count
        self.from_rgb = nn.ModuleList()
        self.channels = channels
        self.latent_size = latent_size

        mul = 2
        inputs = startf
        self.encode_block: nn.ModuleList[DiscriminatorBlock] = nn.ModuleList()

        resolution = 2 ** (self.layer_count + 1)
        # resolution starts from 4x4, hence the plus 1
        resolution_list = []
        # if startf = 32, maxf = 128
        # resolution_list looks like this [[inputs, outputs], [32,64], [64,128], [128,128], ... [128, 128]]
        for i in range(self.layer_count):
            outputs = min(self.maxf, startf * mul)
            resolution_list.append([inputs, outputs])
            self.from_rgb.append(FromRGB(channels, inputs))

            fused_scale = resolution >= 128
            # Call inputs outputs to discriminator block, call dense for last layer
            block = DiscriminatorBlock(
                inputs, outputs, last=False, fused_scale=fused_scale, dense=i == self.layer_count - 1)

            resolution //= 2

            self.encode_block.append(block)
            inputs = outputs
            mul *= 2
        print(resolution_list)
        self.resolution_list = resolution_list

        self.fc2 = ln.Linear(inputs, latent_size, gain=1)

    def encode(self, x, lod):
        x = self.from_rgb[self.layer_count - lod - 1](x)
        print(x.shape)
        x = F.leaky_relu(x, 0.2)

        for i in range(self.layer_count - lod - 1, self.layer_count):
            print(i)
            print(x.shape)
            print(self.resolution_list[i])
            print(self.encode_block[i])
            x = self.encode_block[i](x)

        return self.fc2(x).view(x.shape[0], 1, x.shape[1])

    def encode2(self, x, lod, blend):
        x_orig = x
        x = self.from_rgb[self.layer_count - lod - 1](x)
        x = F.leaky_relu(x, 0.2)
        x = self.encode_block[self.layer_count - lod - 1](x)

        x_prev = F.avg_pool2d(x_orig, 2, 2)

        x_prev = self.from_rgb[self.layer_count - (lod - 1) - 1](x_prev)
        x_prev = F.leaky_relu(x_prev, 0.2)

        x = torch.lerp(x_prev, x, blend)

        for i in range(self.layer_count - (lod - 1) - 1, self.layer_count):
            x = self.encode_block[i](x)

        return self.fc2(x).view(x.shape[0], 1, x.shape[1])

    def forward(self, x, lod, blend):
        if blend == 1:
            return self.encode(x, lod)
        else:
            return self.encode2(x, lod, blend)


@ENCODERS.register("EncoderDefault")
class EncoderDefault(nn.Module):
    def __init__(self, startf, maxf, layer_count, latent_size, channels=3):


        super(EncoderDefault, self).__init__()
        self.maxf = maxf  # maximimum feature maps
        self.startf = startf  # starting feature maps
        # number of layers should be directly proportional to Level Of Detail
        self.layer_count = layer_count
        self.from_rgb: nn.ModuleList[FromRGB] = nn.ModuleList()
        self.channels = channels
        self.latent_size = latent_size

        mul = 2
        inputs = startf
        # every block reduces the height and width by 2
        self.encode_block: nn.ModuleList[EncodeBlock] = nn.ModuleList()

        resolution = 2 ** (self.layer_count + 1)
        resolution_list = []
        for i in range(self.layer_count):
            outputs = min(self.maxf, startf * mul)

            self.from_rgb.append(FromRGB(channels, inputs))

            fused_scale = resolution >= 128

            block = EncodeBlock(inputs, outputs, latent_size,
                                False, fused_scale=fused_scale)

            resolution //= 2

            self.encode_block.append(block)
            inputs = outputs
            mul *= 2
        self.resolution_list = resolution_list

    def encode(self, x, lod):
        styles = torch.zeros(x.shape[0], 1, self.latent_size)

        x = self.from_rgb[self.layer_count - lod - 1](x)
        x = F.leaky_relu(x, 0.2)
        for i in range(self.layer_count - lod - 1, self.layer_count):
            x, s1, s2 = self.encode_block[i](x)
            styles[:, 0] += s1 + s2

        return styles

    def encode2(self, x, lod, blend):
        x_orig = x
        styles = torch.zeros(x.shape[0], 1, self.latent_size)

        x = self.from_rgb[self.layer_count - lod - 1](x)
        x = F.leaky_relu(x, 0.2)

        x, s1, s2 = self.encode_block[self.layer_count - lod - 1](x)
        styles[:, 0] += s1 * blend + s2 * blend

        x_prev = F.avg_pool2d(x_orig, 2, 2)

        x_prev = self.from_rgb[self.layer_count - (lod - 1) - 1](x_prev)
        x_prev = F.leaky_relu(x_prev, 0.2)

        x = torch.lerp(x_prev, x, blend)

        for i in range(self.layer_count - (lod - 1) - 1, self.layer_count):
            # s1 and s2 are in size [batch_size, dlatent_size]
            x, s1, s2 = self.encode_block[i](x)
            styles[:, 0] += s1 + s2

        return styles

    def forward(self, x, lod, blend):
        if blend == 1:
            return self.encode(x, lod)
        else:
            return self.encode2(x, lod, blend)

    def get_statistics(self, lod):
        rgb_std = self.from_rgb[self.layer_count -
                                lod - 1].from_rgb.weight.std().item()
        rgb_std_c = self.from_rgb[self.layer_count - lod - 1].from_rgb.std

        layers = []
        for i in range(self.layer_count - lod - 1, self.layer_count):
            conv_1 = self.encode_block[i].conv_1.weight.std().item()
            conv_1_c = self.encode_block[i].conv_1.std
            conv_2 = self.encode_block[i].conv_2.weight.std().item()
            conv_2_c = self.encode_block[i].conv_2.std
            layers.append(((conv_1 / conv_1_c), (conv_2 / conv_2_c)))
        return rgb_std / rgb_std_c, layers


@GENERATORS.register("GeneratorDefault")
class Generator(nn.Module):
    def __init__(self, startf=32, maxf=256, layer_count=3, latent_size=128, channels=3):
        super(Generator, self).__init__()
        self.maxf = maxf
        self.startf = startf
        self.layer_count = layer_count

        self.channels = channels
        self.latent_size = latent_size

        # multiplier to apply to number of layers for number of feature maps
        mul = 2 ** (self.layer_count - 1)

        # starting inputs
        inputs = min(self.maxf, startf * mul)
        self.const = Parameter(torch.Tensor(1, inputs, 4, 4))
        init.ones_(self.const)

        # [0]*layer_count
        self.layer_to_resolution = [0 for _ in range(layer_count)]
        resolution = 2

        self.style_sizes = []

        to_rgb = nn.ModuleList()

        self.decode_block: nn.ModuleList[DecodeBlock] = nn.ModuleList()
        for i in range(self.layer_count):
            outputs = min(self.maxf, startf * mul)

            has_first_conv = i != 0
            fused_scale = resolution * 2 >= 128

            block = DecodeBlock(inputs, outputs, latent_size,
                                has_first_conv, fused_scale=fused_scale, layer=i)

            # starting resolution 4x4
            resolution *= 2
            self.layer_to_resolution[i] = resolution

            # if first conv use 2*inputs, 2*outputs
            # else 2*outputs, 2*outputs for styles
            self.style_sizes += [2 *
                                 (inputs if has_first_conv else outputs), 2 * outputs]

            to_rgb.append(ToRGB(outputs, channels))

            self.decode_block.append(block)
            inputs = outputs
            mul //= 2

        self.to_rgb = to_rgb

    def decode(self, styles, lod, noise):
        x = self.const

        for i in range(lod + 1):
            x = self.decode_block[i](
                x, styles[:, 2 * i + 0], styles[:, 2 * i + 1], noise)

        x = self.to_rgb[lod](x)
        return x

    def decode2(self, styles, lod, blend, noise):
        x = self.const

        for i in range(lod):
            x = self.decode_block[i](
                x, styles[:, 2 * i + 0], styles[:, 2 * i + 1], noise)

        x_prev = self.to_rgb[lod - 1](x)

        x = self.decode_block[lod](
            x, styles[:, 2 * lod + 0], styles[:, 2 * lod + 1], noise)
        x = self.to_rgb[lod](x)

        needed_resolution = self.layer_to_resolution[lod]

        x_prev = F.interpolate(x_prev, size=needed_resolution)
        x = torch.lerp(x_prev, x, blend)

        return x

    def forward(self, styles, lod, blend, noise):
        if blend == 1:
            return self.decode(styles, lod, noise)
        else:
            return self.decode2(styles, lod, blend, noise)

    def get_statistics(self, lod):
        rgb_std = self.to_rgb[lod].to_rgb.weight.std().item()
        rgb_std_c = self.to_rgb[lod].to_rgb.std

        layers = []
        for i in range(lod + 1):
            conv_1 = 1.0
            conv_1_c = 1.0
            if i != 0:
                conv_1 = self.decode_block[i].conv_1.weight.std().item()
                conv_1_c = self.decode_block[i].conv_1.std
            conv_2 = self.decode_block[i].conv_2.weight.std().item()
            conv_2_c = self.decode_block[i].conv_2.std
            layers.append(((conv_1 / conv_1_c), (conv_2 / conv_2_c)))
        return rgb_std / rgb_std_c, layers





class MappingBlock(nn.Module):
    def __init__(self, n_in, n_out, lrmul) -> None:
        super().__init__()
        self.fc = ln.Linear(n_in, n_out, lrmul=lrmul)

    def forward(self, x):
        x = F.leaky_relu(self.fc(x), 0.2)
        return x


@MAPPINGS.register('MappingD')
class MappingD(nn.Module):
    # Take latent w and predict if its real or fake
    def __init__(self, mapping_layers=5, latent_size=256, dlatent_size=256, mapping_fmaps=256) -> None:
        super().__init__()
        inputs = latent_size
        self.mapping_layers = mapping_layers
        self.map_blocks = nn.ModuleList()
        for i in range(self.mapping_layers):
            outputs = 2 * dlatent_size if i == mapping_layers - 1 else mapping_fmaps
            block = ln.Linear(inputs, outputs, lrmul=0.1)
            inputs = outputs
            self.map_blocks.append(block)
    # TODO Confirm what the output size of this mappingD is

    def forward(self, x):
        for i in range(self.mapping_layers):
            x = self.map_blocks[i](x)

        return x[:, 0, x.shape[2] // 2]


@MAPPINGS.register('MappingF')
class MappingF(nn.Module):
    def __init__(self, num_layers, mapping_layers=5, latent_size=256, dlatent_size=256, mapping_fmaps=256):
        super(MappingF, self).__init__()
        inputs = dlatent_size
        self.mapping_layers = mapping_layers
        self.num_layers = num_layers
        self.map_blocks = nn.ModuleList()
        for i in range(mapping_layers):
            outputs = latent_size if i == mapping_layers-1 else mapping_fmaps
            # not sure what lrmul does, at the surface it seems to provide a multiplier for weight init
            block = MappingBlock(inputs, outputs, lrmul=0.1)
            inputs = outputs
            self.map_blocks.append(block)

    def forward(self, x):
        x = pixel_norm(x)

        for i in range(self.mapping_layers):
            x = self.map_blocks[i](x)

        return x.view(x.shape[0], 1, x.shape[1]).repeat(1, self.num_layers, 1)





# This is not used. Its only for styleGan
class DiscriminatorBlock(nn.Module):
    def __init__(self, inputs, outputs, last=False, fused_scale=True, dense=False):
        """Building Block for Discriminator of StyleGan. For every discriminator block, height and width of feature maps halve.

        Args:
            inputs (int): input channels.
            outputs (int): output channels.
            last (bool, optional): True if this is the last block. Defaults to False.
            fused_scale (bool, optional): True if fused conv2d and upscale2d is used. Defaults to True.
            dense (bool, optional): True if a final layer of dense network is used. Defaults to False.
        """
        super(DiscriminatorBlock, self).__init__()
        self.conv_1 = ln.Conv2d(
            inputs + (1 if last else 0), inputs, 3, 1, 1, bias=False)
        self.bias_1 = nn.Parameter(torch.Tensor(1, inputs, 1, 1))
        self.blur = Blur(inputs)
        self.last = last
        self.dense_ = dense
        self.fused_scale = fused_scale
        if self.dense_:
            # inputs are number of feature maps/channels
            # in the event of last layer, the feature map size 4x4
            # thus the number of inputs for linear layers is feature maps flattened which is inputs x 4 x 4.
            self.dense = ln.Linear(inputs * 4 * 4, outputs)
        else:
            if fused_scale:
                self.conv_2 = ln.Conv2d(
                    inputs, outputs, 3, 2, 1, bias=False, transform_kernel=True)
            else:
                self.conv_2 = ln.Conv2d(inputs, outputs, 3, 1, 1, bias=False)

        self.bias_2 = nn.Parameter(torch.Tensor(1, outputs, 1, 1))

        with torch.no_grad():
            self.bias_1.zero_()
            self.bias_2.zero_()

    def forward(self, x):
        if self.last:
            x = minibatch_stddev_layer(x)

        x = self.conv_1(x) + self.bias_1
        x = F.leaky_relu(x, 0.2)

        if self.dense_:
            x = self.dense(x.view(x.shape[0], -1))
        else:
            x = self.conv_2(self.blur(x))
            if not self.fused_scale:
                x = downscale2d(x)
            x = x + self.bias_2
        x = F.leaky_relu(x, 0.2)

        return x
    
@DISCRIMINATORS.register("DiscriminatorDefault")
class Discriminator(nn.Module):
    def __init__(self, startf=32, maxf=256, layer_count=3, channels=3):
        super(Discriminator, self).__init__()
        self.maxf = maxf
        self.startf = startf
        self.layer_count = layer_count
        self.from_rgb = nn.ModuleList()
        self.channels = channels

        mul = 2
        inputs = startf
        self.encode_block: nn.ModuleList[DiscriminatorBlock] = nn.ModuleList()

        resolution = 2 ** (self.layer_count + 1)

        for i in range(self.layer_count):
            outputs = min(self.maxf, startf * mul)

            self.from_rgb.append(FromRGB(channels, inputs))

            fused_scale = resolution >= 128

            block = DiscriminatorBlock(
                inputs, outputs, i == self.layer_count - 1, fused_scale=fused_scale)

            resolution //= 2

            self.encode_block.append(block)
            inputs = outputs
            mul *= 2

        self.fc2 = ln.Linear(inputs, 1, gain=1)

    def encode(self, x, lod):
        x = self.from_rgb[self.layer_count - lod - 1](x)
        x = F.leaky_relu(x, 0.2)

        for i in range(self.layer_count - lod - 1, self.layer_count):
            x = self.encode_block[i](x)

        return self.fc2(x)

    def encode2(self, x, lod, blend):
        x_orig = x
        x = self.from_rgb[self.layer_count - lod - 1](x)
        x = F.leaky_relu(x, 0.2)
        x = self.encode_block[self.layer_count - lod - 1](x)

        x_prev = F.avg_pool2d(x_orig, 2, 2)

        x_prev = self.from_rgb[self.layer_count - (lod - 1) - 1](x_prev)
        x_prev = F.leaky_relu(x_prev, 0.2)

        x = torch.lerp(x_prev, x, blend)

        for i in range(self.layer_count - (lod - 1) - 1, self.layer_count):
            x = self.encode_block[i](x)
            print(x.shape)
        return self.fc2(x)

    def forward(self, x, lod, blend):
        if blend == 1:
            return self.encode(x, lod)
        else:
            return self.encode2(x, lod, blend)
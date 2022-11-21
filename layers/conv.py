import torch
import torch.nn as nn
import torch.nn.functional as f
import BP.sgbp_ssl as sgbp_ssl


class ConvLayer(nn.Conv3d):
    def __init__(self, network_config, config, name, in_shape, groups=1):
        self.name = name
        self.layer_config = config
        self.network_config = network_config
        self.type = config['type']

        self.batchnorm=config['norm']
        self.normdim=config['out_channels']

        in_features = config['in_channels']
        out_features = config['out_channels']
        kernel_size = config['kernel_size']

        if 'padding' in config:
            padding = config['padding']
        else:
            padding = 0

        if 'stride' in config:
            stride = config['stride']
        else:
            stride = 1

        if 'dilation' in config:
            dilation = config['dilation']
        else:
            dilation = 1

        if 'weight_scale' in config:
            weight_scale = config['weight_scale']
        else:
            weight_scale = 1

        # kernel
        if type(kernel_size) == int:
            kernel = (kernel_size, kernel_size, 1)
        elif len(kernel_size) == 2:
            kernel = (kernel_size[0], kernel_size[1], 1)
        else:
            raise Exception('kernelSize can only be of 1 or 2 dimension. It was: {}'.format(kernel_size.shape))

        # stride
        if type(stride) == int:
            stride = (stride, stride, 1)
        elif len(stride) == 2:
            stride = (stride[0], stride[1], 1)
        else:
            raise Exception('stride can be either int or tuple of size 2. It was: {}'.format(stride.shape))

        # padding
        if type(padding) == int:
            padding = (padding, padding, 0)
        elif len(padding) == 2:
            padding = (padding[0], padding[1], 0)
        else:
            raise Exception('padding can be either int or tuple of size 2. It was: {}'.format(padding.shape))

        # dilation
        if type(dilation) == int:
            dilation = (dilation, dilation, 1)
        elif len(dilation) == 2:
            dilation = (dilation[0], dilation[1], 1)
        else:
            raise Exception('dilation can be either int or tuple of size 2. It was: {}'.format(dilation.shape))

        super(ConvLayer, self).__init__(in_features, out_features, kernel, stride, padding, dilation, groups,
                                        bias=False)
        # nn.init.kaiming_normal_(self.weight)
        self.weight = torch.nn.Parameter(weight_scale * self.weight.cuda(), requires_grad=True)

        self.in_shape = in_shape
        self.out_shape = [out_features, int((in_shape[1]+2*padding[0]-kernel[0])/stride[0]+1),
                          int((in_shape[2]+2*padding[1]-kernel[1])/stride[1]+1)]
        print(self.name)
        print(self.in_shape)
        print(self.out_shape)
        print(list(self.weight.shape))
        print("-----------------------------------------")

    def forward(self, x):
        y=f.conv3d(x, self.weight, self.bias,self.stride, self.padding, self.dilation, self.groups)
        if self.batchnorm==True:
            s=[]
            for t in range(y.shape[-1]):
                s.append(torch.nn.BatchNorm2d(self.normdim).cuda()(y[...,t]))
            y=torch.stack(s,dim=4)

        return y

    def get_parameters(self):
        return self.weight

    def forward_pass(self, x,sg_p):
        y = self.forward(x)  #x是psp  y是对psp进行卷积的结果
        y = sgbp_ssl.SGBP_SSL.apply(y, self.network_config, self.layer_config, sg_p)
        return y

    def weight_clipper(self):
        w = self.weight.data
        w = w.clamp(-4, 4)
        self.weight.data = w

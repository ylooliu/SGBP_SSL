import torch.nn as nn
import layers.conv as conv
import layers.pooling as pooling
import layers.dropout as dropout
import layers.linear as linear
import BP.loss_f as f



class Network(nn.Module):
    def __init__(self, network_config, layers_config, input_shape):
        super(Network, self).__init__()
        self.layers = []
        self.network_config = network_config
        self.layers_config = layers_config

        parameters = []
        print("Network Structure:")
        for key in layers_config:
            c = layers_config[key]

            if c['type'] == 'conv':
                self.layers.append(conv.ConvLayer(network_config, c, key, input_shape))
                input_shape = self.layers[-1].out_shape
                parameters.append(self.layers[-1].get_parameters())
            elif c['type'] == 'linear':
                self.layers.append(linear.LinearLayer(network_config, c, key, input_shape))
                input_shape = self.layers[-1].out_shape
                parameters.append(self.layers[-1].get_parameters())
            elif c['type'] == 'pooling':
                self.layers.append(pooling.PoolLayer(network_config, c, key, input_shape))
                input_shape = self.layers[-1].out_shape
            elif c['type'] == 'dropout':
                self.layers.append(dropout.DropoutLayer(c, key))
            else:
                raise Exception('Undefined layer type. It is: {}'.format(c['type']))
        self.my_parameters = nn.ParameterList(parameters)
        print("-----------------------------------------")

    def forward(self, spike_input, is_train=False,sg_p=None):
        spikes = f.psp(spike_input, self.network_config)
        assert self.network_config['model'] == "LIF"
        
        for l in self.layers:
            if l.type == "dropout":
                if is_train:
                    spikes = l(spikes)
            elif self.network_config["rule"] == "SGBP_SSL":
                spikes = l.forward_pass(spikes,sg_p)   #spikes是返回的本层的突触后电位
            else:
                raise Exception('Unrecognized rule type. It is: {}'.format(self.network_config['rule']))
        return spikes

    def get_parameters(self):
        return self.my_parameters

    def weight_clipper(self):
        for l in self.layers:
            l.weight_clipper()

    def train(self):
        for l in self.layers:
            l.train()

    def eval(self):
        for l in self.layers:
            l.eval()

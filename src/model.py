from torch import nn
import math
import torch as t
import torch.nn.functional as F
from torch_geometric.nn.conv import GraphConv


def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out')  # Kaiming normal initialization
        if m.bias is not None:
            m.bias.data.fill_(0.01)  # Initialize bias with 0.01


class CSGNet(nn.Module):
    def __init__(self, in_channel=1, mid_channel=16, out_channel=1, num_nodes=2207, **args):
        super(CSGNet, self).__init__()
        self.mid_channel = mid_channel
        self.dropout_ratio = args.get('dropout_ratio', 0.2)
        print('Model dropout ratio:', self.dropout_ratio)
        self.num_nodes = num_nodes
        self.global_conv1_dim = args.get('global_conv1_dim', 4 * 3)
        self.global_conv2_dim = args.get('global_conv2_dim', 4)
        self.conv1 = GraphConv(in_channel, mid_channel)
        self.bn1 = nn.LayerNorm((num_nodes, mid_channel))
        self.act1 = nn.ReLU()

        # 2D Convolutional layers
        self.global_conv1 = nn.Conv2d(mid_channel * 1, self.global_conv1_dim, [1, 1])
        self.global_bn1 = nn.BatchNorm2d(self.global_conv1_dim)
        self.global_act1 = nn.ReLU()

        self.global_conv2 = nn.Conv2d(self.global_conv1_dim, self.global_conv2_dim, [1, 1])
        self.global_bn2 = nn.BatchNorm2d(self.global_conv2_dim)
        self.global_act2 = nn.ReLU()

        # Fully connected layers
        last_feature_node = args.get('last_feature_node', 64)
        channel_list = [self.global_conv2_dim * self.num_nodes, 256, last_feature_node]

        if args.get('channel_list', False):
            new_value = self.global_conv2_dim * self.num_nodes
            channel_list = args['channel_list']
            channel_list.insert(0, new_value)
            print("Updated channel_list:", channel_list)
            last_feature_node = channel_list[-1]

        self.nn = []
        for idx, num in enumerate(channel_list[:-1]):
            self.nn.append(nn.Linear(channel_list[idx], channel_list[idx + 1]))
            self.nn.append(nn.BatchNorm1d(channel_list[idx + 1]))
            if self.dropout_ratio > 0:
                self.nn.append(nn.Dropout(0.2))
            self.nn.append(nn.ReLU())

        self.global_fc_nn = nn.Sequential(*self.nn)
        self.fc1 = nn.Linear(last_feature_node, out_channel)

        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        nn.init.kaiming_normal_(self.global_conv1.weight, mode='fan_out')
        nn.init.uniform_(self.global_conv1.bias, -1.0 / math.sqrt(self.mid_channel), 1.0 / math.sqrt(self.mid_channel))

        nn.init.kaiming_normal_(self.global_conv2.weight, mode='fan_out')
        nn.init.uniform_(self.global_conv2.bias, -1.0 / math.sqrt(self.global_conv1_dim), 1.0 / math.sqrt(self.global_conv1_dim))

        # Initialize BatchNorm layers
        nn.init.constant_(self.global_bn1.weight, 1)
        nn.init.constant_(self.global_bn1.bias, 0)
        nn.init.constant_(self.global_bn2.weight, 1)
        nn.init.constant_(self.global_bn2.bias, 0)

        # Initialize fully connected layers
        self.global_fc_nn.apply(init_weights)
        self.fc1.apply(init_weights)

    def forward(self, data, get_latent_varaible=False):
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_weight, data.batch

        # Graph Convolutional layer
        x = self.conv1(x, edge_index=edge_index, edge_weight=edge_weight)
        x = self.act1(x)
        x = x.view(-1, self.num_nodes, self.mid_channel) # (sample, gene_num, embd_dim)
        x = self.bn1(x)
        x = x.permute(1, 0, 2)  # => (gene_num, sample, embd_dim)

        if self.dropout_ratio > 0:
            x = F.dropout(x, p=0.1, training=self.training)

        x = x.permute(1, 2, 0)  # (sample , embd_dim , gene_num)
        x = x.unsqueeze(dim=-1)  # (sample, embd_dim , gene_num , 1)
        x = self.global_conv1(x)  # (sample, embd_dim, gene_num , 1)
        x = self.global_act1(x)
        x = self.global_bn1(x)
        if self.dropout_ratio > 0:
            x = F.dropout(x, p=0.3, training=self.training)
        x = self.global_conv2(x)
        x = self.global_act1(x)
        x = self.global_bn2(x)
        if self.dropout_ratio > 0:
            x = F.dropout(x, p=0.3, training=self.training)
        x = x.squeeze(dim=-1)  # (samples, embd_dim, gene_num)
        num_samples = x.shape[0]

        x = x.view(num_samples, -1)  # flatten
        x = self.global_fc_nn(x)
        latent = x
        x = self.fc1(x)

        if get_latent_varaible:
            return latent, x
        else:
            return x

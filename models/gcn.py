import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    def __init__(self, in_channel, out_channel):
        super(GraphConvolution, self).__init__()
        self.weight = Parameter(torch.FloatTensor(in_channel, out_channel))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.spmm(adj, support)
        return output


class GCN(nn.Module):
    def __init__(self, in_channel, hid_channel, out_channel, dropout=0.5):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(in_channel, hid_channel)
        self.gc2 = GraphConvolution(hid_channel, out_channel)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

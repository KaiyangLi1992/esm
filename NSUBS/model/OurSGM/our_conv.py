from NSUBS.model.OurSGM.config import FLAGS
from NSUBS.model.OurSGM.utils_nn import MLP

import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

from torch_geometric.nn.inits import glorot, zeros


class OurGATConv(MessagePassing):
    def __init__(self,
                 in_channels,
                 out_channels,
                 heads=1,
                 concat=True,
                 negative_slope=0.2,
                 dropout=0,
                 bias=True,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(OurGATConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.weight = Parameter(
            torch.Tensor(in_channels, heads * out_channels))
        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)




        #####
        self.gate_nn = MLP(heads * out_channels, 1, activation_type='elu', num_hidden_lyr=1)
        self.glob_att = MyGlobalAttention(self.gate_nn, None)
        #####

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        zeros(self.bias)

    def forward_gq(self, x1, edge_index1):
        edge_index1, _ = remove_self_loops(edge_index1)
        edge_index1, _ = add_self_loops(edge_index1, num_nodes=x1.size(0))

        x1 = torch.mm(x1, self.weight).view(-1, self.heads, self.out_channels)
        x1 = self.propagate(edge_index1, x=(x1, x1), num_nodes=x1.size(0), K=None)
        return x1

    def forward_gt(self, x2, edge_index2, x1):
        assert edge_index2 is not None
        edge_index2, _ = remove_self_loops(edge_index2)
        edge_index2, _ = add_self_loops(edge_index2, num_nodes=x2.size(0))

        x2 = torch.mm(x2, self.weight).view(-1, self.heads, self.out_channels)

        batch = torch.zeros(x1.shape[0], device=FLAGS.device).long()
        K, _ = self.glob_att(x1, batch) # (1, 32)
        K = K.view(-1, self.heads, self.out_channels) # (1, 4, 8)

        x2 = self.propagate(edge_index2, x=(x2, x2), num_nodes=x2.size(0), K=K)
        # x2 = self.propagate(edge_index2, x=(x2, x2), num_nodes=x2.size(0), K=None)

        return x2

    # def forward(self, x1, edge_index1, x2, edge_index2, x1_cached=None):
    #     if x1_cached is None:
    #         edge_index1, _ = remove_self_loops(edge_index1)
    #         edge_index1, _ = add_self_loops(edge_index1, num_nodes=x1.size(0))
    #
    #         x1 = torch.mm(x1, self.weight).view(-1, self.heads, self.out_channels)
    #         x1 = self.propagate(edge_index1, x=(x1, x1), num_nodes=x1.size(0), K=None)
    #     else:
    #         x1 =x1_cached
    #
    #     if x2 is not None:
    #         assert edge_index2 is not None
    #         edge_index2, _ = remove_self_loops(edge_index2)
    #         edge_index2, _ = add_self_loops(edge_index2, num_nodes=x2.size(0))
    #
    #         x2 = torch.mm(x2, self.weight).view(-1, self.heads, self.out_channels)
    #
    #         batch = torch.zeros(x1.shape[0], device=FLAGS.device).long()
    #         K, _ = self.glob_att(x1, batch) # (1, 32)
    #         K = K.view(-1, self.heads, self.out_channels) # (1, 4, 8)
    #
    #         x2 = self.propagate(edge_index2, x=(x2, x2), num_nodes=x2.size(0), K=K)
    #
    #     return x1, x2

    def message(self, x_i, x_j, edge_index, num_nodes, K): # x_j --> x_i
        # Compute attention coefficients.
        if K is not None:
            # Key change: x_j * K instead of x_j for targer graph (x2).
            K = x_j * K
        else:
            K = x_j
        alpha = (torch.cat([x_i, K], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0], ptr=None, num_nodes=num_nodes)

        # Sample attention coefficients stochastically.
        if self.training and self.dropout > 0:
            alpha = F.dropout(alpha, p=self.dropout, training=True)

        rtn = x_j * alpha.view(-1, self.heads, 1)
        return rtn

    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)





import torch
from torch_scatter import scatter_add
from torch_geometric.utils import softmax

from torch_geometric.nn.inits import reset


class MyGlobalAttention(torch.nn.Module):
    r"""Global soft attention layer from the `"Gated Graph Sequence Neural
    Networks" <https://arxiv.org/abs/1511.05493>`_ paper

    .. math::
        \mathbf{r}_i = \sum_{n=1}^{N_i} \mathrm{softmax} \left(
        h_{\mathrm{gate}} ( \mathbf{x}_n ) \right) \odot
        h_{\mathbf{\Theta}} ( \mathbf{x}_n ),

    where :math:`h_{\mathrm{gate}} \colon \mathbb{R}^F \to
    \mathbb{R}` and :math:`h_{\mathbf{\Theta}}` denote neural networks, *i.e.*
    MLPS.

    Args:
        gate_nn (torch.nn.Module): A neural network :math:`h_{\mathrm{gate}}`
            that computes attention scores by mapping node features :obj:`x` of
            shape :obj:`[-1, in_channels]` to shape :obj:`[-1, 1]`, *e.g.*,
            defined by :class:`torch.nn.Sequential`.
        nn (torch.nn.Module, optional): A neural network
            :math:`h_{\mathbf{\Theta}}` that maps node features :obj:`x` of
            shape :obj:`[-1, in_channels]` to shape :obj:`[-1, out_channels]`
            before combining them with the attention scores, *e.g.*, defined by
            :class:`torch.nn.Sequential`. (default: :obj:`None`)
    """
    def __init__(self, gate_nn, nn=None):
        super(MyGlobalAttention, self).__init__()
        self.gate_nn = gate_nn
        self.nn = nn

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.gate_nn)
        reset(self.nn)

    def forward(self, x, batch, size=None):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        size = batch[-1].item() + 1 if size is None else size

        gate = self.gate_nn(x).view(-1, 1)
        x = self.nn(x) if self.nn is not None else x
        assert gate.dim() == x.dim() and gate.size(0) == x.size(0)

        gate = softmax(gate, batch, num_nodes=size)
        out = scatter_add(gate * x, batch, dim=0, dim_size=size)

        return out, gate

    def __repr__(self):
        return '{}(gate_nn={}, nn={})'.format(self.__class__.__name__,
                                              self.gate_nn, self.nn)


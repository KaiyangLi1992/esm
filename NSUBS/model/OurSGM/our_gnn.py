import torch
import torch.nn.functional as F
from torch_scatter import scatter_add

from NSUBS.model.OurSGM.config import FLAGS

class OurGMN(torch.nn.Module):
    def __init__(self, mlp_att_cross, mlp_value_cross, mlp_att_t, mlp_value_t, mlp_att_q, mlp_value_q):
        super(OurGMN, self).__init__()
        self.mlp_att_cross = mlp_att_cross
        self.mlp_value_cross = mlp_value_cross
        self.mlp_att_t = mlp_att_t
        self.mlp_value_t = mlp_value_t
        self.mlp_att_q = mlp_att_q
        self.mlp_value_q = mlp_value_q

    def forward(self, Xq, edge_index_q, Xt, edge_index_t, u2v_li):
        mask = -torch.ones((Xq.shape[0], Xt.shape[0], Xq.shape[1])) * float('inf')
        for u, v_li in u2v_li.items():
            mask[u,v_li] = 0

        Xcross = torch.cat((Xq.reshape(Xq.shape[0], 1, Xq.shape[1]), Xt.reshape(1, *Xt.shape)), dim=-1)
        att_cross = self.mlp_att_cross(Xcross)
        att_cross = F.softmax(mask.view(*mask.shape, 1) + (att_cross-torch.max(att_cross, dim=0)), dim=0)
        Xqt = torch.sum(att_cross * self.mlp_value_cross(Xcross), dim=0)

        Q = torch.sum(Xqt, dim=0).view(1, -1)
        Xt_merged = torch.cat((Xt, Xqt, Q), dim=1)
        att_t = F.sigmoid(self.mlp_att_t(torch.cat((Xt_merged[edge_index_t[0]],Xt[edge_index_t[1]]), dim=1)))
        Xt = scatter_add(att_t*self.mlp_value_t(Xt_merged[edge_index_t[0]]), edge_index_t)

        att_q = F.sigmoid(self.mlp_att_q(torch.cat((Xq[edge_index_q[0]],Xq[edge_index_q[1]]), dim=1)))
        Xq = scatter_add(att_q*self.mlp_value_q(Xq[edge_index_q[0]]), edge_index_q)

        return Xq, Xt

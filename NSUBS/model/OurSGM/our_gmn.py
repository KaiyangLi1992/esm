from NSUBS.src.utils import OurTimer
import random
import torch
import torch.nn.functional as F
from torch_scatter import scatter_softmax, scatter_add
from torch_geometric.nn import GATv2Conv, GATConv, GCNConv, TransformerConv
# from torch_geometric.nn import  GATConv, GCNConv
from NSUBS.model.OurSGM.config import FLAGS
from NSUBS.model.OurSGM.utils_nn import MLP, get_MLP_args, NormalizeAttention
from yacs.config import CfgNode as CN
from graphgps.layer.gps_layer import GPSLayer

def create_ourgmn_disentangled(dim_in, dim_out, q2t, t2q, gnn_subtype):
    assert q2t and t2q,  'have to implement unidirectional'

    mlp_att_t, mlp_value_t, mlp_att_q, mlp_value_q, gnn_t, gnn_q = \
        None, None, None, None, None, None
    if gnn_subtype == 'ours':
        mlp_att_t = MLP(*get_MLP_args([2 * dim_in, 8, 1]))
        mlp_value_t = MLP(*get_MLP_args([2 * dim_in, dim_out]))
        mlp_att_q = MLP(*get_MLP_args([2 * dim_in, 8, 1]))
        mlp_value_q = MLP(*get_MLP_args([2 * dim_in, dim_out]))
    elif gnn_subtype == 'gatv2':
        gnn_t = GATv2Conv(dim_in, dim_out)
        gnn_q = GATv2Conv(dim_in, dim_out)
    elif gnn_subtype == 'gat':
        gnn_t = GATConv(dim_in, dim_out)
        gnn_q = GATConv(dim_in, dim_out)
    elif gnn_subtype == 'gcn':
        gnn_t = GCNConv(dim_in, dim_out)
        gnn_q = GCNConv(dim_in, dim_out)
    elif gnn_subtype == 'transformer':
        gnn_t = TransformerConv(dim_in, dim_out)
        gnn_q = TransformerConv(dim_in, dim_out)
    elif gnn_subtype == 'graphgps':
        with open(FLAGS.graphgps_config_path, 'r') as f:
            yaml_content = f.read()

        # 使用 load_cfg 方法将 YAML 内容转换为 CfgNode 对象
        cfg = CN.load_cfg(yaml_content)
        local_gnn_type, global_model_type = cfg.gt.layer_type.split('+')


        gnn_t = GPSLayer(dim_h=cfg.gt.dim_hidden,
                local_gnn_type=local_gnn_type,
                global_model_type=global_model_type,
                num_heads=cfg.gt.n_heads,
                act=cfg.gnn.act,
                pna_degrees=cfg.gt.pna_degrees,
                equivstable_pe=cfg.posenc_EquivStableLapPE.enable,
                dropout=cfg.gt.dropout,
                attn_dropout=cfg.gt.attn_dropout,
                layer_norm=cfg.gt.layer_norm,
                batch_norm=cfg.gt.batch_norm,
                bigbird_cfg=cfg.gt.bigbird,
                log_attn_weights=cfg.train.mode == 'log-attn-weights',
            )
        
        gnn_q = GPSLayer(dim_h=cfg.gt.dim_hidden,
                local_gnn_type=local_gnn_type,
                global_model_type=global_model_type,
                num_heads=cfg.gt.n_heads,
                act=cfg.gnn.act,
                pna_degrees=cfg.gt.pna_degrees,
                equivstable_pe=cfg.posenc_EquivStableLapPE.enable,
                dropout=cfg.gt.dropout,
                attn_dropout=cfg.gt.attn_dropout,
                layer_norm=cfg.gt.layer_norm,
                batch_norm=cfg.gt.batch_norm,
                bigbird_cfg=cfg.gt.bigbird,
                log_attn_weights=cfg.train.mode == 'log-attn-weights',
            )


  


    gmn_intra = OurGMNCustomIntra(gnn_subtype, gnn_t, gnn_q, mlp_att_t, mlp_value_t, mlp_att_q, mlp_value_q)

    mlp_att_cross_q = MLP(*get_MLP_args([dim_out, dim_out]))
    mlp_att_cross_t = MLP(*get_MLP_args([dim_out, dim_out]))
    mlp_val_cross_q = MLP(*get_MLP_args([dim_out, dim_out]))
    mlp_merge_t = MLP(*get_MLP_args([3 * dim_out, dim_out]))
    mlp_val_cross_t = MLP(*get_MLP_args([dim_out, dim_out]))
    mlp_merge_q = MLP(*get_MLP_args([2 * dim_out, dim_out]))
    gmn_inter = \
        OurGMNCustomInter(
            mlp_att_cross_q, mlp_att_cross_t,
            mlp_val_cross_q, mlp_val_cross_t,
            mlp_merge_q, mlp_merge_t
        )

    our_gmn = OurGMNCustomWrapper(gmn_inter, gmn_intra)
    return our_gmn

def create_ourgmn(dim_in, dim_out, q2t, t2q):
    mlp_att_cross_q = MLP(*get_MLP_args([dim_in, dim_in]))
    mlp_att_cross_t = MLP(*get_MLP_args([dim_in, dim_in]))
    if q2t:
        mlp_val_cross_q = MLP(*get_MLP_args([dim_in, dim_in]))
        mlp_merge_t = MLP(*get_MLP_args([3 * dim_in, dim_in]))
    else:
        mlp_val_cross_q = None
        mlp_merge_t = None
    mlp_att_t = MLP(*get_MLP_args([2 * dim_in, 8, 1]))
    mlp_value_t = MLP(*get_MLP_args([2 * dim_in, dim_out]))
    if t2q:
        mlp_val_cross_t = MLP(*get_MLP_args([dim_in, dim_in]))
        mlp_merge_q = MLP(*get_MLP_args([2 * dim_in, dim_in]))
    else:
        mlp_val_cross_t = None
        mlp_merge_q = None
    mlp_att_q = MLP(*get_MLP_args([2 * dim_in, 8, 1]))
    mlp_value_q = MLP(*get_MLP_args([2 * dim_in, dim_out]))
    our_gmn = \
        OurGMNCustom(
            mlp_att_cross_q, mlp_att_cross_t, mlp_val_cross_q, mlp_val_cross_t,
            mlp_att_t, mlp_value_t, mlp_att_q, mlp_value_q, mlp_merge_q, mlp_merge_t,
            q2t, t2q
        )
    return our_gmn


class OurGMNCustomWrapper(torch.nn.Module):
    def __init__(self, gmn_inter, gmn_intra):
        super(OurGMNCustomWrapper, self).__init__()
        self.gmn_inter = gmn_inter
        self.gmn_intra = gmn_intra

    def forward(self, pyg_data_q, pyg_data_t,
                  norm_q, norm_t, u2v_li, node_mask,
                  only_run_inter=True):
        if only_run_inter:
            pyg_data_q.x, pyg_data_t.x = self.gmn_inter(pyg_data_q.x, pyg_data_t.x, u2v_li)
            return pyg_data_q,pyg_data_t
        else:
            return self.gmn_intra(pyg_data_q, pyg_data_t)


class OurGMNCustomIntra(torch.nn.Module):
    def __init__(self, gnn_subtype, gnn_t, gnn_q, mlp_att_t, mlp_value_t, mlp_att_q, mlp_value_q):
        super(OurGMNCustomIntra, self).__init__()
        self.gnn_subtype = gnn_subtype
        self.gnn_t = gnn_t
        self.gnn_q = gnn_q
        self.mlp_att_t = mlp_att_t
        self.mlp_value_t = mlp_value_t
        self.mlp_att_q = mlp_att_q
        self.mlp_value_q = mlp_value_q

    def forward(self, pyg_data_q, pyg_data_t):
        # if self.gnn_subtype == 'ours':
        #     msg_q = torch.cat((Xq[edge_index_q[0]], Xq[edge_index_q[1]]), dim=1)
        #     Xq = \
        #         scatter_add(
        #             scatter_softmax(self.mlp_att_q(msg_q), edge_index_q[1], dim=0) * \
        #             self.mlp_value_q(msg_q),
        #             edge_index_q[1],
        #             dim=0, dim_size=Xq.shape[0]
        #         )  # / norm_q

        #     msg_t = torch.cat((Xt[edge_index_t[0]], Xt[edge_index_t[1]]), dim=1)
        #     Xt = \
        #         scatter_add(
        #             scatter_softmax(self.mlp_att_t(msg_t), edge_index_t[1], dim=0) * \
        #             self.mlp_value_t(msg_t),
        #             edge_index_t[1],
        #             dim=0, dim_size=Xt.shape[0]
        #         )  # / norm_t
        # else:
        pyg_data_q = self.gnn_q(pyg_data_q)
        pyg_data_t= self.gnn_t(pyg_data_t)
        return pyg_data_q, pyg_data_t

class OurGMNCustomInter(torch.nn.Module):
    def __init__(self, mlp_att_cross_q, mlp_att_cross_t, mlp_val_cross_q, mlp_val_cross_t,
                 mlp_merge_q, mlp_merge_t):
        super(OurGMNCustomInter, self).__init__()
        self.mlp_att_cross_q = mlp_att_cross_q
        self.mlp_att_cross_t = mlp_att_cross_t
        self.mlp_val_cross_q = mlp_val_cross_q
        self.mlp_val_cross_t = mlp_val_cross_t
        self.mlp_merge_q = mlp_merge_q
        self.mlp_merge_t = mlp_merge_t


    def forward(self, Xq, Xt, u2v_li):
        '''
        Note: node_mask is essentially masked out because the edge_index does not include
        incoming edges from masked nodes, and u2v_li doesn't include node_masked nodes by definition
        '''
        assert not FLAGS.use_node_mask_diameter, print('check if above is true with this flag^')
        Xq_att_cross = F.elu(self.mlp_att_cross_q(Xq))
        Xt_att_cross = F.elu(self.mlp_att_cross_t(Xt))

        att_logits, u_li_cat, v_li_cat = [], [], []

        # timer = None
        # if FLAGS.time_analysis:
        #     timer = OurTimer()

        for u, v_li in u2v_li.items():

            # if FLAGS.time_analysis:
            #     timer.time_and_clear(f'u={u} begin')

            if FLAGS.k_sample_cross_graph is not None and \
                    len(v_li) > FLAGS.k_sample_cross_graph:
                # print(f'sampled: {len(v_li)} -> {FLAGS.k_sample_cross_graph}!')
                v_li_sample = random.sample(v_li, k=FLAGS.k_sample_cross_graph)

                # if FLAGS.time_analysis:
                #     timer.time_and_clear(f'u={u} sampled!!!!!')
            else:
                v_li_sample = v_li

                # if FLAGS.time_analysis:
                #     timer.time_and_clear(f'u={u} quick')
            u_li_cat.extend([u] * len(v_li_sample))
            v_li_cat.extend(v_li_sample)
            att_logits.append(
                torch.matmul(
                    Xq_att_cross[u], Xt_att_cross[v_li_sample].transpose(0, 1)
                ).view(-1)
            )

            # if FLAGS.time_analysis:
            #     timer.time_and_clear(f'u={u} att_logits.append')
        att_logits = torch.cat(att_logits, dim=0)

        # if FLAGS.time_analysis:
        #     timer.time_and_clear(f'end att_logits')
        #     timer.print_durations_log()

        # print('Xt_val_cross[u_li_cat].shape', Xt_val_cross[u_li_cat].shape)
        Xt_val_cross = F.elu(self.mlp_val_cross_t(Xt))
        cross_index_v2u = torch.tensor(u_li_cat, dtype=torch.long, device=FLAGS.device)
        att_u = scatter_softmax(att_logits.view(-1), cross_index_v2u).view(-1, 1)
        Xt2q = \
            scatter_add(
                att_u * Xt_val_cross[u_li_cat],
                cross_index_v2u, dim=0, dim_size=Xq.shape[0]
            )
        # T = torch.mean(Xt, dim=0).view(1, Xt.shape[1]).expand(Xt.shape[0], Xq.shape[1])
        Xq_merged = self.mlp_merge_q(torch.cat((Xq, Xt2q), dim=1))

        Xq_val_cross = F.elu(self.mlp_val_cross_q(Xq))
        cross_index_u2v = torch.tensor(v_li_cat, dtype=torch.long, device=FLAGS.device)
        att_v = scatter_softmax(att_logits.view(-1), cross_index_u2v).view(-1, 1)
        Xq2t = \
            scatter_add(
                att_v * Xq_val_cross[u_li_cat],
                cross_index_u2v, dim=0, dim_size=Xt.shape[0]
            )
        Q = torch.mean(Xq, dim=0).view(1, Xq.shape[1]).expand(Xt.shape[0], Xq.shape[1])
        Xt_merged = self.mlp_merge_t(torch.cat((Xt, Xq2t, Q), dim=1))

        return Xq_merged, Xt_merged

class OurGMNCustom(torch.nn.Module):
    def __init__(self, mlp_att_cross_q, mlp_att_cross_t, mlp_val_cross_q, mlp_val_cross_t,
                 mlp_att_t, mlp_value_t, mlp_att_q, mlp_value_q, mlp_merge_q, mlp_merge_t,
                 q2t, t2q):
        super(OurGMNCustom, self).__init__()
        self.mlp_att_cross_q = mlp_att_cross_q
        self.mlp_att_cross_t = mlp_att_cross_t
        self.mlp_val_cross_q = mlp_val_cross_q
        self.mlp_val_cross_t = mlp_val_cross_t
        self.mlp_att_t = mlp_att_t
        self.mlp_value_t = mlp_value_t
        self.mlp_att_q = mlp_att_q
        self.mlp_value_q = mlp_value_q
        self.mlp_merge_q = mlp_merge_q
        self.mlp_merge_t = mlp_merge_t
        self.q2t = q2t
        self.t2q = t2q
        # self.norm_q = NormalizeAttention()
        # self.norm_t = NormalizeAttention()

    def forward(self, Xq, edge_index_q, Xt, edge_index_t, norm_q, norm_t, u2v_li, node_mask,
                only_run_inter=True):
        '''
        Note: node_mask is essentially masked out because the edge_index does not include
        incoming edges from masked nodes, and u2v_li doesn't include node_masked nodes by definition
        '''
        assert not FLAGS.use_node_mask_diameter, print('check if above is true with this flag^')
        Xq_att_cross = F.elu(self.mlp_att_cross_q(Xq))
        Xt_att_cross = F.elu(self.mlp_att_cross_t(Xt))

        att_logits, u_li_cat, v_li_cat = [], [], []

        # timer = None
        # if FLAGS.time_analysis:
        #     timer = OurTimer()

        for u, v_li in u2v_li.items():

            # if FLAGS.time_analysis:
            #     timer.time_and_clear(f'u={u} begin')

            if FLAGS.k_sample_cross_graph is not None and \
                    len(v_li) > FLAGS.k_sample_cross_graph:
                # print(f'sampled: {len(v_li)} -> {FLAGS.k_sample_cross_graph}!')
                v_li_sample = random.sample(v_li, k=FLAGS.k_sample_cross_graph)

                # if FLAGS.time_analysis:
                #     timer.time_and_clear(f'u={u} sampled!!!!!')

            else:
                v_li_sample = v_li

                # if FLAGS.time_analysis:
                #     timer.time_and_clear(f'u={u} quick')
            u_li_cat.extend([u] * len(v_li_sample))
            v_li_cat.extend(v_li_sample)
            att_logits.append(
                torch.matmul(
                    Xq_att_cross[u], Xt_att_cross[v_li_sample].transpose(0, 1)
                ).view(-1)
            )

            # if FLAGS.time_analysis:
            #     timer.time_and_clear(f'u={u} att_logits.append')
        att_logits = torch.cat(att_logits, dim=0)

        # if FLAGS.time_analysis:
        #     timer.time_and_clear(f'end att_logits')
        #     timer.print_durations_log()

        if self.t2q:
            # print('Xt_val_cross[u_li_cat].shape', Xt_val_cross[u_li_cat].shape)
            Xt_val_cross = F.elu(self.mlp_val_cross_t(Xt))
            cross_index_v2u = torch.tensor(u_li_cat, dtype=torch.long, device=FLAGS.device)
            att_u = scatter_softmax(att_logits.view(-1), cross_index_v2u).view(-1, 1)
            Xt2q = \
                scatter_add(
                    att_u * Xt_val_cross[u_li_cat],
                    cross_index_v2u, dim=0, dim_size=Xq.shape[0]
                )
            # T = torch.mean(Xt, dim=0).view(1, Xt.shape[1]).expand(Xt.shape[0], Xq.shape[1])
            Xq_merged = self.mlp_merge_q(torch.cat((Xq, Xt2q), dim=1))
        else:
            Xq_merged = Xq

        if self.q2t:
            Xq_val_cross = F.elu(self.mlp_val_cross_q(Xq))
            cross_index_u2v = torch.tensor(v_li_cat, dtype=torch.long, device=FLAGS.device)
            att_v = scatter_softmax(att_logits.view(-1), cross_index_u2v).view(-1, 1)
            Xq2t = \
                scatter_add(
                    att_v * Xq_val_cross[u_li_cat],
                    cross_index_u2v, dim=0, dim_size=Xt.shape[0]
                )
            Q = torch.mean(Xq, dim=0).view(1, Xq.shape[1]).expand(Xt.shape[0], Xq.shape[1])
            Xt_merged = self.mlp_merge_t(torch.cat((Xt, Xq2t, Q), dim=1))
        else:
            Xt_merged = Xt

        if only_run_inter:
            # |Gq| x 2|D| -> |Gq| x Dout
            Xq = Xq_merged  # self.mlp_inter_out_q(Xq_merged)
            Xt = Xt_merged  # self.mlp_inter_out_t(Xt_merged)
        else:
            # |Gt(E)| x 6|D| -> |Gt(E)| x Dout
            msg_t = torch.cat((Xt_merged[edge_index_t[0]], Xt_merged[edge_index_t[1]]), dim=1)
            Xt = \
                scatter_add(
                    scatter_softmax(self.mlp_att_t(msg_t), edge_index_t[1], dim=0) * \
                    self.mlp_value_t(msg_t),
                    edge_index_t[1],
                    dim=0, dim_size=Xt.shape[0]
                )  # / norm_t

            msg_q = torch.cat((Xq_merged[edge_index_q[0]], Xq_merged[edge_index_q[1]]), dim=1)
            Xq = \
                scatter_add(
                    scatter_softmax(self.mlp_att_q(msg_q), edge_index_q[1], dim=0) * \
                    self.mlp_value_q(msg_q),
                    edge_index_q[1],
                    dim=0, dim_size=Xq.shape[0]
                )  # / norm_q

        return Xq, Xt

# def create_ourgmn_old(dim_in, dim_out):
#     mlp_att_cross_q = MLP(*get_MLP_args([dim_in, dim_in]))
#     mlp_att_cross_t = MLP(*get_MLP_args([dim_in, dim_in]))
#     mlp_val_cross_q = MLP(*get_MLP_args([dim_in, dim_in]))
#     mlp_att_t = MLP(*get_MLP_args([4*dim_in, 1]))
#     mlp_value_t = MLP(*get_MLP_args([4*dim_in, dim_out]))
#     mlp_att_q = MLP(*get_MLP_args([2*dim_in, 1]))
#     mlp_value_q = MLP(*get_MLP_args([2*dim_in, dim_out]))
#     return OurGMNOld(mlp_att_cross_q, mlp_att_cross_t, mlp_val_cross_q, mlp_att_t, mlp_value_t, mlp_att_q, mlp_value_q)
#
# class OurGMNOld(torch.nn.Module):
#     def __init__(self, mlp_att_cross_q, mlp_att_cross_t, mlp_val_cross_q, mlp_att_t, mlp_value_t, mlp_att_q, mlp_value_q):
#         super(OurGMNOld, self).__init__()
#         self.mlp_att_cross_q = mlp_att_cross_q
#         self.mlp_att_cross_t = mlp_att_cross_t
#         self.mlp_val_cross_q = mlp_val_cross_q
#         self.mlp_att_t = mlp_att_t
#         self.mlp_value_t = mlp_value_t
#         self.mlp_att_q = mlp_att_q
#         self.mlp_value_q = mlp_value_q
#
#     def forward(self, Xq, edge_index_q, Xt, edge_index_t, norm_q, norm_t, u2v_li, node_mask):
#         Xq_att_cross = F.elu(self.mlp_att_cross_q(Xq))
#         Xq_val_cross = F.elu(self.mlp_val_cross_q(Xq))
#         Xt_att_cross = F.elu(self.mlp_att_cross_t(Xt))
#
#         att_logits, u_li_cat, v_li_cat = [], [], []
#         for u, v_li in u2v_li.items():
#             u_li_cat.extend([u]*len(v_li))
#             v_li_cat.extend(v_li)
#             att_logits.append(
#                 torch.matmul(
#                     Xq_att_cross[u], Xt_att_cross[v_li].transpose(0,1)
#                 ).view(-1)
#             )
#         cross_index = torch.tensor(v_li_cat, dtype=torch.long, device=FLAGS.device)
#
#         att_logits = torch.cat(att_logits, dim=0)
#         att = scatter_softmax(att_logits.view(-1), cross_index).view(-1,1)
#         Xqt = \
#             scatter_add(
#                 att * Xq_val_cross[u_li_cat],
#                 cross_index, dim=0, dim_size=Xt.shape[0]
#             )
#
#         # Xcross = \
#         #     torch.cat((
#         #         Xq.reshape(Xq.shape[0], 1, Xq.shape[1]).expand(Xq.shape[0], *Xt.shape),
#         #         Xt.reshape(1, *Xt.shape).expand(Xq.shape[0], *Xt.shape)
#         #     ), dim=-1)
#         #
#         # att_cross_logits = self.mlp_att_cross(Xcross)
#         # att_cross_masked = -torch.ones((Xq.shape[0], Xt.shape[0], 1), dtype=torch.float32, device=FLAGS.device) * float('inf')
#         # for u, v_li in u2v_li.items():
#         #     att_cross_masked[u, v_li] = att_cross_logits[u, v_li]
#         # att_cross_masked = att_cross_masked - att_cross_masked.max(dim=0)[0]
#         # att_cross = torch.exp(att_cross_masked)/torch.sum(torch.exp(att_cross_masked), dim=0)
#         # att_cross[:, node_mask, :] = 0
#         # if att_cross.isinf().any() or att_cross.isnan().any():
#         #     print(att_cross.isinf().any(), att_cross.isnan().any())
#         #     assert False
#         #
#         # Xqt = torch.sum(att_cross * self.mlp_value_cross(Xcross), dim=0)
#
#         Q = torch.mean(Xq, dim=0).view(1, Xq.shape[1]).expand(Xt.shape[0], Xq.shape[1])
#
#         Xt_merged = torch.cat((Xt, Xqt, Q), dim=1)
#         msg_t = torch.cat((Xt_merged[edge_index_t[0]],Xt[edge_index_t[1]]), dim=1)
#         att_t = F.sigmoid(self.mlp_att_t(msg_t))
#         Xt = \
#             scatter_add(
#                 att_t*self.mlp_value_t(msg_t),
#                 edge_index_t[1],
#                 dim=0, dim_size=Xt.shape[0]
#             ) / norm_t
#
#         msg_q = torch.cat((Xq[edge_index_q[0]],Xq[edge_index_q[1]]), dim=1)
#         att_q = F.sigmoid(self.mlp_att_q(msg_q))
#         Xq = \
#             scatter_add(
#                 att_q*self.mlp_value_q(msg_q),
#                 edge_index_q[1],
#                 dim=0, dim_size=Xq.shape[0]
#             ) / norm_q
#
#         return Xq, Xt


import torch.nn as nn
from torch_geometric.nn import EdgeConv

class GMNPropagator(nn.Module):
    def __init__(self, input_dim, output_dim, more_nn='None', distance_metric='cosine',
                 f_node='MLP'):
        super().__init__()
        self.out_dim = output_dim
        if distance_metric == 'cosine':
            self.distance_metric = nn.CosineSimilarity()
        elif distance_metric == 'euclidean':
            self.distance_metric = nn.PairwiseDistance()
        self.softmax = nn.Softmax(dim=1)
        self.f_messasge = MLP(2 * input_dim, 2 * input_dim, num_hidden_lyr=1, hidden_channels=[
            2 * input_dim])  # 2*input_dim because in_dim = dim(g1) + dim(g2)
        self.f_node_name = f_node
        if f_node == 'MLP':
            self.f_node = MLP(4 * input_dim, output_dim,
                              num_hidden_lyr=1)  # 2*input_dim for m_sum, 1 * input_dim for u_sum and 1*input_dim for x
        elif f_node == 'GRU':
            self.f_node = nn.GRUCell(3 * input_dim,
                                     input_dim)  # 2*input_dim for m_sum, 1 * input_dim for u_sum
        else:
            raise ValueError("{} for f_node has not been implemented".format(f_node))
        self.more_nn = more_nn
        if more_nn == 'None':
            pass
        elif more_nn == 'EdgeConv':
            nnl = nn.Sequential(nn.Linear(2 * input_dim, output_dim), nn.ReLU(),
                                nn.Linear(output_dim, output_dim))
            self.more_conv = EdgeConv(nnl, aggr='max')
            self.proj_back = nn.Sequential(nn.Linear(2 * output_dim, output_dim), nn.ReLU(),
                                           nn.Linear(output_dim, output_dim))
        else:
            raise ValueError("{} has not been implemented".format(more_nn))

    def __call__(self, x1, edge_index1, x2, edge_index2,
                  norm_q, norm_t, u2v_li, node_mask,
                  only_run_inter=True):
        x = torch.cat([x1, x2], dim=0)
        M, N = x1.shape[0], x2.shape[0]
        edge_index = torch.cat([edge_index1, edge_index2 + M], dim=1)
        row, col = edge_index
        m = torch.cat((x[row], x[col]), dim=1)  # E by (2 * D)
        m = self.f_messasge(m)
        m_sum = scatter_add(m, row, dim=0, dim_size=x.size(0))  # N(gs) by (2 * D)
        u_sum = self.f_match(x1, x2)  # u_sum has shape N(gs) by D

        if self.f_node_name == 'MLP':
            in_f_node = torch.cat((x, m_sum, u_sum), dim=1)
            out = self.f_node(in_f_node)
        elif self.f_node_name == 'GRU':
            in_f_node = torch.cat((m_sum, u_sum), dim=1)  # N by 3*D
            out = self.f_node(in_f_node, x)

        if self.more_nn != 'None':
            more_out = self.more_conv(x, edge_index)
            # Concat the GMN output with the additional output.
            out = torch.cat((out, more_out), dim=1)
            out = self.proj_back(out)  # back to output_dim

        return out[:x1.shape[0]], out[x1.shape[0]:]

    def f_match(self, x1, x2):  # x, batch_data):
        '''from the paper https://openreview.net/pdf?id=S1xiOjC9F7'''
        u_all_l = []

        u1 = self._f_match_helper(x1, x2)  # N(g1) by D tensor
        u2 = self._f_match_helper(x2, x1)  # N(g2) by D tensor

        u_all_l.append(u1)
        u_all_l.append(u2)

        return torch.cat(u_all_l, dim=0)

    def _f_match_helper(self, g1x, g2x):
        g1_norm = torch.nn.functional.normalize(g1x, p=2, dim=1)
        g2_norm = torch.nn.functional.normalize(g2x, p=2, dim=1)
        g1_sim = torch.matmul(g1_norm, torch.t(g2_norm))

        # N_1 by N_2 tensor where a1[x][y] is the softmaxed a_ij of the yth node of g2 to the xth node of g1
        a1 = self.softmax(g1_sim)

        sum_a1_h = torch.sum(g2x * a1[:, :, None],
                             dim=1)  # N1 by D tensor where each row is sum_j(a_j * h_j)
        return g1x - sum_a1_h


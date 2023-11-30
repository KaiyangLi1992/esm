import sys
# sys.path.append('/home/kli16/NSUBS/src/')
from NSUBS.src.utils import OurTimer
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv, GATConv, JumpingKnowledge
from collections import defaultdict

from NSUBS.model.OurSGM.our_conv import OurGATConv
from NSUBS.model.OurSGM.our_gmn import create_ourgmn, create_ourgmn_disentangled, GMNPropagator
from NSUBS.model.OurSGM.dvn_decoder import SimilarityConcatDecoder, SimilarityBilinearDecoder
from NSUBS.model.OurSGM.config import FLAGS
from NSUBS.model.OurSGM.utils_nn import MLP, get_MLP_args

def create_encoder(d_in):
    assert FLAGS.dvn_config['encoder']['type'] == 'GNNConsensusEncoder'
    def get_gnn(dim1, dim2):
        if FLAGS.dvn_config['encoder']['gnn_type'] == 'GAT':
            gnn = GATConv(dim1, dim2)
        elif FLAGS.dvn_config['encoder']['gnn_type'] == 'GCN':
            gnn = GCNConv(dim1, dim2)
        elif FLAGS.dvn_config['encoder']['gnn_type'] == 'OurGAT':
            gnn = OurGATConv(dim1, dim2)
        else:
            assert False
        return gnn

    def get_gnn_pair(din, dout):
        if FLAGS.dvn_config['encoder']['gnn_type'] == 'OurGAT':
            gnn = OurGATConv(din, dout)
            out = [gnn, None, None]
        elif FLAGS.dvn_config['encoder']['gnn_type'] == 'OurGMN':
            gnn = \
                create_ourgmn(
                    din, dout, FLAGS.dvn_config['encoder']['q2t'], FLAGS.dvn_config['encoder']['t2q'])
            out = [gnn, None, None]
        elif FLAGS.dvn_config['encoder']['gnn_type'] == 'OurGMNv2':
            gnn = \
                create_ourgmn_disentangled(
                    din, dout,
                    FLAGS.dvn_config['encoder']['q2t'],
                    FLAGS.dvn_config['encoder']['t2q'],
                    FLAGS.dvn_config['encoder']['gnn_subtype']
                )
            out = [gnn, None, None]
        elif FLAGS.dvn_config['encoder']['gnn_type'] == 'GMN':
            gnn = GMNPropagator(din, dout)
            out = [gnn, None, None]
        elif FLAGS.dvn_config['encoder']['shared_gnn_weights']:
            gnn = get_gnn(din, dout)
            out = [None, gnn, gnn]
        else:
            gnn1, gnn2 = get_gnn(din, dout), get_gnn(din, dout)
            out = [None, gnn1, gnn2]
        return out

    # create encoder gnn
    gnn_dims = [d_in] + FLAGS.dvn_config['encoder']['hidden_gnn_dims']
    gnn_wrapper_li = \
        torch.nn.ModuleList([
            GNNWrapper(*get_gnn_pair(din, dout)) \
            for (din, dout) in zip(gnn_dims, gnn_dims[1:])
        ])
    assert FLAGS.dvn_config['encoder']['consensus_cfg_li'] is None
    consensus_li = [None]*len(gnn_wrapper_li)
    encoder_gnn_consensus = GNNConsensusEncoder(gnn_wrapper_li, consensus_li)
    return encoder_gnn_consensus, gnn_dims[-1]

def create_consensus(intergraph_interact_cfg):
    interact_type = intergraph_interact_cfg['type']
    if interact_type == 'bilinearv2':
        mlp_in_dims = intergraph_interact_cfg['mlp_in_dims']
        mlp_out_dims = intergraph_interact_cfg['mlp_out_dims']

        mlp_in_alpha = MLP(*get_MLP_args(mlp_in_dims))
        mlp_out_alpha = MLP(*get_MLP_args(mlp_out_dims))
        interact_bilinear_alpha = \
            SimilarityBilinearDecoder(mlp_in_alpha, mlp_out_alpha, mlp_in_dims[-1], mlp_out_dims[0])

        mlp_in_beta = MLP(*get_MLP_args(mlp_in_dims))
        mlp_out_beta = MLP(*get_MLP_args(mlp_out_dims))
        interact_bilinear_beta  = \
            SimilarityBilinearDecoder(mlp_in_beta, mlp_out_beta, mlp_in_dims[-1], mlp_out_dims[0])

        consensus = IntergraphInteractV2(interact_type, interact_bilinear_alpha, interact_bilinear_beta)
    elif interact_type == 'mlpv2':
        mlp_in_dims = intergraph_interact_cfg['mlp_in_dims']
        mlp_out_dims = intergraph_interact_cfg['mlp_out_dims']

        mlp_in_alpha = MLP(*get_MLP_args(mlp_in_dims))
        mlp_out_alpha = MLP(*get_MLP_args(mlp_out_dims))
        interact_mlp_alpha = \
            SimilarityConcatDecoder(mlp_in_alpha, mlp_out_alpha)

        mlp_in_beta = MLP(*get_MLP_args(mlp_in_dims))
        mlp_out_beta = MLP(*get_MLP_args(mlp_out_dims))
        interact_mlp_beta = \
            SimilarityConcatDecoder(mlp_in_beta, mlp_out_beta)

        consensus = IntergraphInteractV2(interact_type, interact_mlp_alpha, interact_mlp_beta)
    elif interact_type == 'bilinear':
        mlp_in_dims = intergraph_interact_cfg['mlp_in_dims']
        mlp_out_dims = intergraph_interact_cfg['mlp_out_dims']
        mlp_in = MLP(*get_MLP_args(mlp_in_dims))
        mlp_out = MLP(*get_MLP_args(mlp_out_dims))
        interact_bilinear = \
            SimilarityBilinearDecoder(mlp_in, mlp_out, mlp_in_dims[-1], mlp_out_dims[0])
        consensus = IntergraphInteract(interact_type, interact_bilinear=interact_bilinear)
    elif interact_type == 'mlp':
        mlp_in_dims = intergraph_interact_cfg['mlp_in_dims']
        mlp_out_dims = intergraph_interact_cfg['mlp_out_dims']
        mlp_in = MLP(*get_MLP_args(mlp_in_dims))
        mlp_out = MLP(*get_MLP_args(mlp_out_dims))
        assert mlp_in_dims[-1]*2 == mlp_out_dims[0]
        interact_mlp = SimilarityConcatDecoder(mlp_in, mlp_out)
        consensus = IntergraphInteract(interact_type, interact_mlp=interact_mlp)
    elif interact_type == 'fixed_point_wise':
        interact_coeff = intergraph_interact_cfg['coeff']
        consensus = IntergraphInteract(interact_type, interact_coeff=interact_coeff)
    elif interact_type == 'fixed_mean_wise':
        interact_coeff = intergraph_interact_cfg['coeff']
        consensus = IntergraphInteract(interact_type, interact_coeff=interact_coeff)
    else:
        assert False
    return consensus

class GNNWrapper(torch.nn.Module):
    def __init__(self, gnnm, gnnq, gnnt):
        super(GNNWrapper, self).__init__()
        self.gnnm = gnnm
        self.gnnq = gnnq
        self.gnnt = gnnt

    def forward(self, Xq,  edge_indexq, Xt, edge_indext, norm_q, norm_t, u2v_li, node_mask, only_inter):
        assert self.gnnm is not None, f'{self.gnnm} {self.gnnq} {self.gnnt}'
        if self.gnnm is None:
            Xq = self.gnnq(Xq, edge_indexq)
            Xt = self.gnnt(Xt, edge_indext)
        else:
            if type(self.gnnm) is OurGATConv:
                Xq = self.gnnm.forward_gq(Xq, edge_indexq)
                Xt = self.gnnm.forward_gt(Xt, edge_indext, Xq)
            else:
                Xq, Xt = self.gnnm(Xq, edge_indexq, Xt, edge_indext, norm_q, norm_t, u2v_li, node_mask, only_inter)
        return Xq, Xt

class GNNConsensusEncoder(torch.nn.Module):
    def __init__(self, gnn_wrapper_li, consensus_li):
        super(GNNConsensusEncoder, self).__init__()
        self.gnn_wrapper_li = gnn_wrapper_li
        self.consensus_li = consensus_li
        self.jk = JumpingKnowledge('max')
        self.reset_cache()

    def reset_cache(self):
        self.Xq_Xt_cached_li = None

    def forward(self, Xq, edge_indexq, Xt, edge_indext,
                nn_map, cs_map, candidate_map,
                norm_q, norm_t, u2v_li, node_mask,
                cache_embeddings):
        timer = None
        if FLAGS.time_analysis:
            timer = OurTimer()

        # if cache_embeddings:
        Xq_li, Xt_li = [Xq], [Xt]

        for i, (gnn_wrapper) in \
                enumerate(self.gnn_wrapper_li):
            # Xq, Xt = Xq_Xt_cached
            Xq, Xt = gnn_wrapper(Xq, edge_indexq, Xt, edge_indext, norm_q, norm_t, u2v_li, node_mask, only_inter=True)
            Xt[node_mask, :] = 0
            if i != len(self.gnn_wrapper_li)-1:
                Xq, Xt = F.elu(Xq), F.elu(Xt)
            Xq_li.append(Xq)
            Xt_li.append(Xt)

            # if FLAGS.time_analysis:
            #     timer.time_and_clear(f'gnn_wrapper 2')

            Xq = self.jk(Xq_li)
            Xt = self.jk(Xt_li)

            # if FLAGS.time_analysis:
            #     timer.time_and_clear(f'jk')
        else:
            Xq_li, Xt_li = [Xq], [Xt]
            for i, (gnn_wrapper, consensus) in \
                    enumerate(zip(self.gnn_wrapper_li, self.consensus_li)):
                Xq, Xt = gnn_wrapper(Xq, edge_indexq, Xt, edge_indext, norm_q, norm_t, u2v_li, node_mask, only_inter=False)
                if i != len(self.gnn_wrapper_li)-1:
                    Xq, Xt = F.elu(Xq), F.elu(Xt)
                Xt[node_mask, :] = 0
                if consensus is not None:
                    Xq, Xt = consensus(Xq, Xt, nn_map, cs_map, candidate_map)
                Xq_li.append(Xq)
                Xt_li.append(Xt)
            Xq = self.jk(Xq_li)
            Xt = self.jk(Xt_li)

            if FLAGS.time_analysis:
                timer.time_and_clear(f'gnn no cache')

        if FLAGS.time_analysis:
            timer.print_durations_log()
        return Xq, Xt

class IntergraphInteractV2(torch.nn.Module):
    def __init__(self, interact_type, interact_alpha, interact_beta):
        super(IntergraphInteractV2, self).__init__()
        assert interact_type in ['mlpv2', 'bilinearv2'] # check Xq_in in forward if add another type
        self.interact_type = interact_type
        self.interact_alpha = interact_alpha
        self.interact_beta = interact_beta

    def forward(self, Xq, Xt, nn_map, cs_map, candidate_map):
        if len(nn_map) > 0:
            u_consensus_li, v_consensus_li = np.array(
                [[int(u), int(v)] for (u, v) in nn_map.items()]).T
            Xt[v_consensus_li] = Xq[u_consensus_li]

        v_candidates = set()
        u2v_li, v2u_li = {}, defaultdict(list)
        for u in set(cs_map.keys()) - set(nn_map.keys()):
            v_li = candidate_map[u] if (candidate_map is not None and u in candidate_map) else cs_map[u]
            v_li = list(set(v_li) - set(nn_map.values()))
            u2v_li[u] = v_li
            # for v in v_li: v2u_li[v].append(u)
            v_candidates.update(set(v_li))

        v_candidates = list(v_candidates)
        v2v_candidates = -torch.ones(Xt.shape[0], dtype=torch.long, device=FLAGS.device)
        v2v_candidates[v_candidates] = torch.arange(len(v_candidates), dtype=torch.long, device=FLAGS.device)

        a = torch.zeros((len(v_candidates), Xq.shape[0]), dtype=torch.float, device=FLAGS.device) - float('inf') # softmax_scores
        h = torch.zeros((len(v_candidates), Xq.shape[0], Xq.shape[1]), dtype=torch.float, device=FLAGS.device) # embeddings
        for u, v_li in u2v_li.items():
            Xq_in = Xq[[u]].repeat(len(v_li), 1) if self.interact_type == 'mlpv2' else Xq[[u]]
            Xt_in = Xt[v_li]
            beta = F.sigmoid(self.interact_beta(Xq_in, Xt_in)).view(-1, 1)
            alpha = F.elu(self.interact_alpha(Xq_in, Xt_in)).view(-1, 1)
            h[v2v_candidates[v_li], u] = ((1-beta)*Xq[[u]]+beta*Xt[v_li])
            a[v2v_candidates[v_li], u] = alpha.reshape(len(v_li))

        a = torch.exp(a - torch.max(a, dim=1)[0].reshape(a.shape[0], 1))
        if a.isinf().any() or a.isnan().any() or h.isinf().any() or h.isnan().any():
            print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
            print(f'v_candidates: {min(v_candidates)} {max(v_candidates)} {len(v_candidates)} {Xt.shape}, {Xq.shape}')
            print(f'a: {a.min()} {a.max()} {a.isinf().any()} {a.isnan().any()} {a.shape}')
            print(f'a: {h.min()} {h.max()} {h.isinf().any()} {h.isnan().any()} {h.shape}')
            print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
            assert False
        a = a/torch.sum(a, dim=1).reshape(a.shape[0], 1) + 1e-10
        if a.isinf().any() or a.isnan().any() or h.isinf().any() or h.isnan().any():
            print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
            print(f'v_candidates: {min(v_candidates)} {max(v_candidates)} {len(v_candidates)} {Xt.shape}, {Xq.shape}')
            print(f'a: {a.min()} {a.max()} {a.isinf().any()} {a.isnan().any()} {a.shape}')
            print(f'a: {h.min()} {h.max()} {h.isinf().any()} {h.isnan().any()} {h.shape}')
            print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
            assert False

        a = a.reshape(*a.shape, 1)
        Xt[v_candidates] = torch.sum(a*h, dim=1)
        return Xq, Xt

class IntergraphInteract(torch.nn.Module):
    def __init__(self, interact_type, interact_bilinear=None, interact_mlp=None, interact_coeff=None):
        super(IntergraphInteract, self).__init__()
        self.interact_type = interact_type
        self.interact_bilinear = interact_bilinear
        self.interact_mlp = interact_mlp
        self.interact_coeff = interact_coeff

    def forward(self, Xq, Xt, nn_map, cs_map, candidate_map):
        norm_vec = torch.ones(Xt.shape[0], device=FLAGS.device)
        for u, v_li in cs_map.items():
            if u not in nn_map.keys():
                norm_vec[list(set(v_li)-set(nn_map.values()))] += 1 # size of unmatched nodes in cs partition
        norm_vec = norm_vec.view(-1,1)

        if self.interact_type == 'bilinear':
            norm_vec = torch.zeros(Xt.shape[0], device=FLAGS.device) + 1e-10
            Xt_unnorm = torch.zeros_like(Xt)
            for u, v_li in cs_map.items():
                if u not in nn_map.keys():
                    v_candidates = list(set(v_li) - set(nn_map.values()))
                    matching_matrix = self.interact_bilinear(Xq[[u]], Xt[v_candidates])
                    beta = F.sigmoid(matching_matrix).view(-1).repeat(Xq.shape[1],1).transpose(0,1)
                    beta_softmax = torch.exp(matching_matrix).view(-1)
                    norm_vec[v_candidates] += beta_softmax
                    beta_softmax = beta_softmax.repeat(Xq.shape[1],1).transpose(0,1)
                    Xt_unnorm[v_candidates] += beta_softmax * ((1 - beta) * Xq[[u]] + beta * Xt[v_candidates])
            Xt = Xt_unnorm/(norm_vec.view(-1,1))
        elif self.interact_type == 'mlp':
            norm_vec = torch.zeros(Xt.shape[0], device=FLAGS.device) + 1e-10
            Xt_unnorm = torch.zeros_like(Xt)
            for u, v_li in cs_map.items():
                if u not in nn_map.keys():
                    v_candidates = list(set(v_li) - set(nn_map.values()))
                    matching_matrix = self.interact_mlp(Xq[u].repeat(len(v_candidates), 1), Xt[v_candidates])
                    beta = F.sigmoid(matching_matrix).view(-1).repeat(Xq.shape[1],1).transpose(0,1)
                    beta_softmax = torch.exp(matching_matrix).view(-1)
                    norm_vec[v_candidates] += beta_softmax
                    beta_softmax = beta_softmax.repeat(Xq.shape[1],1).transpose(0,1)
                    Xt_unnorm[v_candidates] += beta_softmax * ((1 - beta) * Xq[[u]] + beta * Xt[v_candidates])
            Xt = Xt_unnorm/(norm_vec.view(-1,1))
        elif self.interact_type == 'fixed_point_wise':
            Xt = norm_vec * FLAGS.nudge_factor * Xt
            for u, v_li in cs_map.items():
                if u not in nn_map.keys():
                    Xt[list(set(v_li)-set(nn_map.values()))] += (1-self.interact_coeff) * Xq[u]
            Xt = Xt/norm_vec
        elif self.interact_type == 'fixed_mean_wise':
            Xt = norm_vec * Xt
            for u, v_li in cs_map.items():
                if u not in nn_map.keys():
                    v_candidates = list(set(v_li)-set(nn_map.values()))
                    Xt[v_candidates] += \
                        (self.interact_coeff - 1) * torch.mean(Xt[v_candidates], dim=0) + \
                        (1 - self.interact_coeff) * Xq[u]
            Xt = Xt/norm_vec
        elif self.interact_type is None or self.interact_type == 'None':
            pass
        else:
            assert False

        if len(nn_map) > 0:
            u_consensus_li, v_consensus_li = np.array(
                [[int(u), int(v)] for (u, v) in nn_map.items()]).T
            Xt[v_consensus_li] = Xq[u_consensus_li]

        return Xq, Xt
    
def main():
    gnn,_ = create_encoder(5)
    Xq = torch.randn(5, 2)
    Xt = torch.randn(5, 2)
    edge_indexq = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])  # 模拟的边
    edge_indext = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])



    # 这些是`forward`函数需要的其他参数
    nn_map, cs_map, candidate_map, norm_q, norm_t, u2v_li, node_mask = None, None, None, None, None, None, None
    cache_embeddings = True

    Xq_out, Xt_out = gnn(Xq, edge_indexq, Xt, \
                              edge_indext, nn_map, cs_map, \
                                candidate_map, norm_q, norm_t, \
                                    u2v_li, node_mask, cache_embeddings)

if __name__ == '__main__':
    main()

import torch
import torch.nn.functional as F
import torch.nn as nn
import networkx as nx
import numpy as np
import random

from NSUBS.model.OurSGM.config import FLAGS
from NSUBS.model.OurSGM.utils_nn import MLP, get_MLP_args, NormalizeAttention
from NSUBS.model.OurSGM.utils_our import get_accum_reward_from_start

def create_decoder():
    decoder_policy = create_decoder_policy()
    decoder_value = create_decoder_dvn()
    return decoder_policy, decoder_value

def create_decoder_policy():
    assert FLAGS.dvn_config['decoder_policy']['type'] in ['GLSearch', 'bilinear_custom', 'bilinear', 'concat']
    if FLAGS.dvn_config['decoder_policy']['type'] == 'bilinear_custom':
        mlp_in_dims = FLAGS.dvn_config['decoder_policy']['similarity_decoder']['mlp_in_dims']
        mlp_out_dims = FLAGS.dvn_config['decoder_policy']['similarity_decoder']['mlp_out_dims'].copy()
        g_emb_dim = FLAGS.dvn_config['decoder_policy']['similarity_decoder']['g_emb']
        mlp_out_dims[0] += g_emb_dim
        mlp_in = MLP(*get_MLP_args(mlp_in_dims))
        mlp_out = MLP(*get_MLP_args(mlp_out_dims))
        similarity_decoder = \
            BilinearDecoder(mlp_in, mlp_out, mlp_in_dims[-1], mlp_out_dims[0]-g_emb_dim)
    elif FLAGS.dvn_config['decoder_policy']['type'] == 'GLSearch':
        similarity_decoder = \
            GLSearchDecoderWrapper(FLAGS.dvn_config['decoder_policy']['similarity_decoder']['mlp_in_dims'][0])
    elif FLAGS.dvn_config['decoder_policy']['type'] == 'bilinear':
        mlp_in_dims = FLAGS.dvn_config['decoder_policy']['similarity_decoder']['mlp_in_dims']
        mlp_out_dims = FLAGS.dvn_config['decoder_policy']['similarity_decoder']['mlp_out_dims']
        mlp_in = MLP(*get_MLP_args(mlp_in_dims))
        mlp_out = MLP(*get_MLP_args(mlp_out_dims))
        similarity_decoder = \
            SimilarityBilinearDecoder(mlp_in, mlp_out, mlp_in_dims[-1], mlp_out_dims[0])
    elif FLAGS.dvn_config['decoder_policy']['type'] == 'concat':
        mlp_in_dims = FLAGS.dvn_config['decoder_policy']['similarity_decoder']['mlp_in_dims']
        mlp_out_dims = FLAGS.dvn_config['decoder_policy']['similarity_decoder']['mlp_out_dims']
        mlp_in = MLP(*get_MLP_args(mlp_in_dims))
        mlp_out = MLP(*get_MLP_args(mlp_out_dims))
        assert mlp_in_dims[-1]*2 == mlp_out_dims[0]
        similarity_decoder = \
            SimilarityConcatDecoder(mlp_in, mlp_out)
    else:
        assert False
    return similarity_decoder

def create_decoder_dvn():
    assert FLAGS.dvn_config['decoder_dvn']['type'] in ['Simple', 'Query'], print('didnt implement throwaway phi(|nn_map|)')
    if FLAGS.dvn_config['decoder_dvn']['type'] == 'Query':
        mlp_att = MLP(*get_MLP_args(FLAGS.dvn_config['decoder_dvn']['simple_decoder']['mlp_att']))
        mlp_val = MLP(*get_MLP_args(FLAGS.dvn_config['decoder_dvn']['simple_decoder']['mlp_val']))
        mlp_final = MLP(*get_MLP_args(FLAGS.dvn_config['decoder_dvn']['simple_decoder']['mlp_final']))
        decoder_wl_similarity = QueryDecoder(mlp_att, mlp_val, mlp_final)
    elif FLAGS.dvn_config['decoder_dvn']['type'] == 'Simple':
        mlp_att_q = MLP(*get_MLP_args(FLAGS.dvn_config['decoder_dvn']['simple_decoder']['mlp_att_q']))
        mlp_val_q = MLP(*get_MLP_args(FLAGS.dvn_config['decoder_dvn']['simple_decoder']['mlp_val_q']))
        mlp_att_t = MLP(*get_MLP_args(FLAGS.dvn_config['decoder_dvn']['simple_decoder']['mlp_att_t']))
        mlp_val_t = MLP(*get_MLP_args(FLAGS.dvn_config['decoder_dvn']['simple_decoder']['mlp_val_t']))
        mlp_final = MLP(*get_MLP_args(FLAGS.dvn_config['decoder_dvn']['simple_decoder']['mlp_final']))
        decoder_wl_similarity = SimpleDecoder(mlp_att_q, mlp_val_q, mlp_att_t, mlp_val_t, mlp_final)
    elif FLAGS.dvn_config['decoder_dvn']['type'] == 'WLSimilarityDecoder':
        # get dimensions of MLPs
        mlp_coarsen_dims = FLAGS.dvn_config['decoder_dvn']['graph_encoder']['mlp_dims']
        mlp_query_dims = FLAGS.dvn_config['decoder_dvn']['graph_encoder_proj']['mlp_query_dims']
        mlp_key_dims = FLAGS.dvn_config['decoder_dvn']['graph_encoder_proj']['mlp_key_dims']
        assert mlp_coarsen_dims[0] == mlp_query_dims[0] == mlp_key_dims[0]

        # create all MLPs
        mlp_coarsen = MLP(*get_MLP_args(mlp_coarsen_dims))
        mlp_query = MLP(*get_MLP_args(mlp_query_dims))
        mlp_key = MLP(*get_MLP_args(mlp_key_dims))
        if FLAGS.dvn_config['decoder_dvn']['shared_graph_encoder_weights']:
            mlp_value = mlp_coarsen
        else:
            mlp_value_dims = FLAGS.dvn_config['decoder_dvn']['graph_encoder_proj']['mlp_value_dims']
            assert mlp_value_dims[-1] == mlp_coarsen_dims[-1] and mlp_value_dims[0] == mlp_coarsen_dims[0]
            mlp_value = MLP(*get_MLP_args(mlp_value_dims))

        # create decoder blocks
        graph_encoder = GraphEncoder(mlp_coarsen)
        projected_graph_encoder = \
            ProjectedGraphEncoder(
                TransformerAttention(mlp_query, mlp_key, mlp_value, FLAGS.dvn_config['decoder_dvn']['graph_encoder_proj']['temp'])
            )

        if FLAGS.dvn_config['decoder_dvn']['coarsen_function']['type'] == 'sum':
            coarsen_function = SumCoarsener(FLAGS.dvn_config['decoder_dvn']['coarsen_function']['use_mapped_embeddings'])
        elif FLAGS.dvn_config['decoder_dvn']['coarsen_function']['type'] == 'subgraph_connected_unconnected':
            dim = mlp_coarsen_dims[-1]
            coarsen_function = SubgraphConnectedUnconnectedCoarsener(dim)
        else:
            assert False

        if FLAGS.dvn_config['decoder_dvn']['similarity_decoder']['type'] == 'bilinear':
            mlp_in_dims = FLAGS.dvn_config['decoder_dvn']['similarity_decoder']['mlp_in_dims']
            mlp_out_dims = FLAGS.dvn_config['decoder_dvn']['similarity_decoder']['mlp_out_dims']
            mlp_in = MLP(*get_MLP_args(mlp_in_dims))
            mlp_out = MLP(*get_MLP_args(mlp_out_dims))
            similarity_decoder = \
                SimilarityBilinearDecoder(mlp_in, mlp_out, mlp_in_dims[-1], mlp_out_dims[0])
        elif FLAGS.dvn_config['decoder_dvn']['similarity_decoder']['type'] == 'concat':
            mlp_in_dims = FLAGS.dvn_config['decoder_dvn']['similarity_decoder']['mlp_in_dims']
            mlp_out_dims = FLAGS.dvn_config['decoder_dvn']['similarity_decoder']['mlp_out_dims']
            mlp_in = MLP(*get_MLP_args(mlp_in_dims))
            mlp_out = MLP(*get_MLP_args(mlp_out_dims))
            assert mlp_in_dims[-1]*2 == mlp_out_dims[0]
            similarity_decoder = \
                SimilarityConcatDecoder(mlp_in, mlp_out)
        elif FLAGS.dvn_config['decoder_dvn']['similarity_decoder']['type'] == 'partition':
            assert FLAGS.dvn_config['decoder_dvn']['coarsen_function']['type'] == 'subgraph_connected_unconnected'
            partition_out_dim = 0
            partition_func_li = torch.nn.ModuleList()
            assert(len(FLAGS.dvn_config['decoder_dvn']['similarity_decoder']['partition_func_li']) == 3)
            for partition_args in FLAGS.dvn_config['decoder_dvn']['similarity_decoder']['partition_func_li']:
                if partition_args['type'] == 'sum':
                    partition_func_li.append(SimilaritySimpleSumDecoder())
                    partition_out_dim += mlp_coarsen_dims[-1]
                elif partition_args['type'] == 'concat':
                    mlp_in_dims = partition_args['mlp_in_dims']
                    mlp_out_dims = partition_args['mlp_out_dims']
                    mlp_in = MLP(*get_MLP_args(mlp_in_dims))
                    mlp_out = MLP(*get_MLP_args(mlp_out_dims))
                    assert mlp_in_dims[0] == mlp_coarsen_dims[-1]
                    assert mlp_in_dims[-1]*2 == mlp_out_dims[0]
                    partition_func_li.append(SimilarityConcatDecoder(mlp_in, mlp_out))
                    partition_out_dim += mlp_out_dims[-1]
                else:
                    assert False

            mlp_sim_decoder_li = FLAGS.dvn_config['decoder_dvn']['similarity_decoder']['mlp_decoder_dims']
            assert partition_out_dim == mlp_sim_decoder_li[0]
            partition_decoder = MLP(*get_MLP_args(mlp_sim_decoder_li))

            dim = mlp_coarsen_dims[-1]

            similarity_decoder = \
                SimilarityPartitionDecoder(dim, partition_func_li, partition_decoder)
        else:
            assert False

        # create decoder
        decoder_wl_similarity = \
            WLSimilarityDecoder(graph_encoder, projected_graph_encoder, coarsen_function, similarity_decoder)
    else:
        assert False

    return decoder_wl_similarity


class BilinearDecoder(torch.nn.Module):
    def __init__(self, mlp_in, mlp_out, d_in, d_out):
        super(BilinearDecoder, self).__init__()
        self.encoder = mlp_in
        self.bilinear_mat = torch.nn.Parameter(torch.randn(d_in, d_in, d_out) + torch.eye(d_in).view(d_in, d_in, 1))
        self.decoder = mlp_out

    def forward(self, Xq, Xt, u, v_li, gq, gt, nn_map, cs_map, candidate_map, execute_action, query_tree, g_emb):
        Xsgq_latent, Xsgt_latent = self.encoder(Xq[[u]]), self.encoder(Xt[v_li])
        sim_latent = torch.einsum('ik,klj,hl->ihj', Xsgq_latent, self.bilinear_mat, Xsgt_latent)
        sim = self.decoder(torch.cat((sim_latent, g_emb.view(1,1,-1).repeat(*sim_latent.shape[:-1], 1)), dim=2))
        return sim, sim_latent.squeeze(dim=0)


class GLSearchDecoderWrapper(torch.nn.Module):
    def __init__(self, d):
        super(GLSearchDecoderWrapper, self).__init__()
        self.decoder = GLSearchDecoder(d)

    def forward(self, Xq, Xt, u, v_li, gq, gt, nn_map, cs_map, candidate_map, execute_action, query_tree, g_emb):
        if execute_action is None and query_tree is None:
            out, _ = self.decoder(Xq, Xt, gq, gt, nn_map, candidate_map, None, None)
        else:
            idx_v_li_sample = random.sample(list(enumerate(v_li)), k=min(20, len(v_li)))
            v_li_sample = []
            idx_li_sample = []
            for idx, value in idx_v_li_sample:
                v_li_sample.append(value)
                idx_li_sample.append(idx)
            out = torch.zeros(len(v_li), dtype=torch.float32, device=FLAGS.device)

            sim = []
            nn_map_uv_li, candidate_map_li = \
                execute_action_li(u, v_li_sample, gq, gt, nn_map, cs_map, execute_action, query_tree)
            for i, (nn_map, candidate_map) in enumerate(zip(nn_map_uv_li, candidate_map_li)):
                # print(f'run_nn: {i}/{len(candidate_map_li)}')
                sim.append(self.decoder(Xq, Xt, gq, gt, nn_map, candidate_map, None, None)[0])
            sim = torch.stack(sim, dim=0).view(-1)
            out[idx_li_sample] = sim - sim.min() + 1e-12
        return out, None

from copy import deepcopy
from NSUBS.model.OurSGM.data_structures_common import GlobalSearchParams
class HackyWrapper():
    def __init__(self, nx_graph):
        self.nx_graph = nx_graph

def execute_action_li(u, v_li, gq, gt, nn_map, cs_map, execute_action, query_tree):
    v_li_valid, nn_map_uv_li, candidate_map_li = [], [], []
    for i,v in enumerate(v_li):
        # print(f'exec_action: {i}/{len(v_li)}')
        nn_map_uv = deepcopy(nn_map)
        nn_map_uv[u] = v
        global_search_params = \
            GlobalSearchParams(
                HackyWrapper(gq), HackyWrapper(gt), (cs_map, query_tree), None, None, None,
                None, None, None, None, None, None
            )
        candidate_map = execute_action(global_search_params, nn_map_uv)
        nn_map_uv_li.append(nn_map_uv)
        candidate_map_li.append(candidate_map)
    return nn_map_uv_li, candidate_map_li


class GLSearchInteract(torch.nn.Module):
    def __init__(self, d):
        super(GLSearchInteract, self).__init__()
        self.CNN = \
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=(3,), stride=(1,), padding=1)

    def forward(self, x1, x2):
        if len(x1.shape) == 1:
            x1 = x1.reshape(1,-1)
        if len(x2.shape) == 1:
            x2 = x2.reshape(1,-1)
        z1 = self.CNN(x1.unsqueeze(1)).view(x1.shape[0], -1)
        z2 = self.CNN(x2.unsqueeze(1)).view(x1.shape[0], -1)
        Z = torch.max(torch.stack((z1, z2), dim=0), dim=0)[0]
        return Z

class GLSearchReadout(torch.nn.Module):
    def __init__(self, d):
        super(GLSearchReadout, self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(d, d),
            torch.nn.ELU(),
            torch.nn.Linear(d, d),
        )

    def forward(self, X):
        Z = X.sum(dim=0)
        return self.mlp(Z)

class GLSearchDecoder(torch.nn.Module):
    def __init__(self, d):
        super(GLSearchDecoder, self).__init__()
        self.readout_g1 = GLSearchReadout(d)
        self.readout_g2 = GLSearchReadout(d)
        self.interact_g = GLSearchInteract(d)

        self.readout_s1 = GLSearchReadout(d)
        self.readout_s2 = GLSearchReadout(d)
        self.interact_s = GLSearchInteract(d)

        self.readout_d1 = GLSearchReadout(d)
        self.readout_d2 = GLSearchReadout(d)
        self.interact_d = GLSearchInteract(d)
        self.readout_dc = GLSearchReadout(d)

        self.readout_d1 = GLSearchReadout(d)
        self.readout_d2 = GLSearchReadout(d)
        self.interact_d = GLSearchInteract(d)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(4*d, d),
            torch.nn.ELU(),
            torch.nn.Linear(d, d//2),
            torch.nn.ELU(),
            torch.nn.Linear(d//2, d//4),
            torch.nn.ELU(),
            torch.nn.Linear(d//4, 1)
        )

    def forward(self, Xq, Xt, gq, gt, nn_map, candidate_map, node_mask, logger):
        g1 = self.readout_g1(Xq)
        g2 = self.readout_g2(Xt)
        g = self.interact_g(g1, g2)

        s1 = self.readout_s1(Xq[list(nn_map.keys())])
        s2 = self.readout_s2(Xt[list(nn_map.values())])
        s = self.interact_s(s1, s2)

        if len(candidate_map) == 0:
            dc = torch.zeros_like(s)
        else:
            dc = []
            for u, v_li in candidate_map.items():
                dj1 = self.readout_d1(Xq[[u]])
                dj2 = self.readout_d2(Xt[v_li])
                dc.append(self.interact_d(dj1, dj2))
            dc = self.readout_dc(torch.stack(dc, dim=0))

        query_unconnected = list(set(gq.nodes) - set(candidate_map.keys()) - set(nn_map.keys()))
        target_unconnected = list(set(gt.nodes) - set().union(*[set(v_li) for v_li in candidate_map.values()]) - set(nn_map.values()))
        d01 = self.readout_d1(Xq[query_unconnected]) if len(query_unconnected) > 0 else torch.zeros_like(s)
        d02 = self.readout_d2(Xt[target_unconnected]) if len(target_unconnected) > 0 else torch.zeros_like(s)
        d0 = self.interact_d(d01, d02)

        Q = torch.cat([g, s, dc, d0], dim=-1)
        V = self.mlp(Q)
        return V, Q

class QueryDecoder(torch.nn.Module):
    def __init__(self, mlp_att, mlp_val, mlp_final):
        super(QueryDecoder, self).__init__()
        self.mlp_att = mlp_att
        self.mlp_val = mlp_val
        self.mlp_final = mlp_final
        self.norm = NormalizeAttention()

    def forward(self, Xq, Xt, gq, gt, nn_map, candidate_map, node_mask, logger):
        Q = torch.sum(self.norm(self.mlp_att(Xq)).view(-1, 1) * self.mlp_val(Xq), dim=0) / Xq.shape[0]
        V = F.leaky_relu(self.mlp_final(Q)) - get_accum_reward_from_start(len(nn_map), gq.number_of_nodes(), False)
        return V, Q

class SimpleDecoder(torch.nn.Module):
    def __init__(self, mlp_att_q, mlp_val_q, mlp_att_t, mlp_val_t, mlp_final):
        super(SimpleDecoder, self).__init__()
        self.mlp_att_q = mlp_att_q
        self.mlp_val_q = mlp_val_q
        self.mlp_att_t = mlp_att_t
        self.mlp_val_t = mlp_val_t
        self.mlp_final = mlp_final

    def forward(self, Xq, Xt, gq, gt, nn_map, candidate_map, node_mask, logger):
        # Xq = Xq[list(set(gq.nodes()) - set(nn_map.keys()))]
        Xt = Xt[list(set(gt.nodes()) - set(node_mask))] #- set(nn_map.values()) - set(node_mask))]
        Q = torch.sum(F.sigmoid(self.mlp_att_q(Xq)).view(-1,1) * self.mlp_val_q(Xq), dim=0)/Xq.shape[0]
        T = torch.sum(F.sigmoid(self.mlp_att_t(Xt)).view(-1,1) * self.mlp_val_t(Xt), dim=0)/Xt.shape[0]
        G = torch.cat((Q, T), dim=-1)
        V = self.mlp_final(G)
        return V, G

class WLSimilarityDecoder(torch.nn.Module):
    def __init__(self, graph_encoder, projected_graph_encoder, coarsen_function, similarity_decoder):
        super(WLSimilarityDecoder, self).__init__()
        self.graph_encoder = graph_encoder
        self.projected_graph_encoder = projected_graph_encoder
        self.coarsen_function = coarsen_function
        self.similarity_decoder = similarity_decoder

    def forward(self, Xq, Xt, gq, gt, nn_map, candidate_map, logger):
        nidq_mapped = list(nn_map.keys())
        nidt_mapped = list(nn_map.values())
        nidq_unmapped = list(candidate_map.keys())

        # candidate_map must have all u indices in the disjoint union
        # otherwise, the Xsg embedding for target graph will be missing entries!
        assert Xq.shape[0] != len(nidq_mapped)
        # assert set(range(Xq.shape[0])) - set(nidq_mapped) == set(nidq_unmapped)

        X_mapped = self.graph_encoder((Xq[nidq_mapped]+Xt[nidt_mapped])/2)
        Xq_unmapped = self.graph_encoder(Xq[nidq_unmapped])
        Xt_unmapped, u2i = self.projected_graph_encoder(Xq, Xt, candidate_map)

        Xsgq = self.coarsen_function(X_mapped, Xq_unmapped, nidq_mapped, nidq_unmapped, gq)
        Xsgt = self.coarsen_function(X_mapped, Xt_unmapped, nidq_mapped, nidq_unmapped, gq, u2i)

        v = self.similarity_decoder(Xsgq, Xsgt, logger)

        return v


class GraphEncoder(torch.nn.Module):
    def __init__(self, mlp):
        super(GraphEncoder, self).__init__()
        self.encoder = mlp

    def forward(self, X):
        Xsg = self.encoder(X)
        return Xsg

class ProjectedGraphEncoder(torch.nn.Module):
    def __init__(self, attention_mechanism):
        super(ProjectedGraphEncoder, self).__init__()
        self.attention_mechanism = attention_mechanism

    def forward(self, Xq, Xt, candidate_map):
        assert len(candidate_map) > 0

        Xsg_li, u2i = [], {}
        for i, (u, v_li) in enumerate(candidate_map.items()):
            Xsg_li.append(self.attention_mechanism(query=Xq[u], key=Xt[v_li], value=Xt[v_li]))
            u2i[u] = i

        Xsg = torch.stack(Xsg_li, dim=0)
        return Xsg, u2i

class SumCoarsener(torch.nn.Module):
    def __init__(self, use_mapped_embeddings):
        super(SumCoarsener, self).__init__()
        self.use_mapped_embeddings = use_mapped_embeddings

    def forward(self, X_mapped, X_unmapped, *other):
        if self.use_mapped_embeddings:
            Xsg = torch.sum(X_mapped, dim=0) + torch.sum(X_unmapped, dim=0)
        else:
            Xsg = torch.sum(X_unmapped, dim=0)
        return Xsg

class SubgraphConnectedUnconnectedCoarsener(torch.nn.Module):
    def __init__(self, d):
        super(SubgraphConnectedUnconnectedCoarsener, self).__init__()
        self.X_map_bias = torch.nn.Parameter(torch.randn(1, d, requires_grad=True)/d)
        self.X_connected_bias = torch.nn.Parameter(torch.randn(1, d, requires_grad=True)/d)
        self.X_unconnected_bias = torch.nn.Parameter(torch.randn(1, d, requires_grad=True)/d)

    def forward(self, X_mapped, X_unmapped, sg_nids, candidate_nids, g, u2i=None):
        if u2i is None:
            assert type(candidate_nids) == list
            u2i = {u:i for (i, u) in  enumerate(candidate_nids)}
        candidate_nids_connected = self._intersect_neighbors_of_set1_with_set2(set(sg_nids), set(candidate_nids), g)
        candidate_nids_unconnected = list(set(candidate_nids) - set(candidate_nids_connected))

        assert set(u2i.keys()) == set(candidate_nids), print(set(u2i.keys()), '\n', candidate_nids)
        candidate_nids_connected = [u2i[u] for u in candidate_nids_connected]
        candidate_nids_unconnected = [u2i[u] for u in candidate_nids_unconnected]

        X = \
            torch.cat([
                self.X_map_bias+torch.sum(X_mapped, dim=0).view(1,-1),
                self.X_connected_bias+torch.sum(X_unmapped[candidate_nids_connected], dim=0).view(1,-1),
                self.X_unconnected_bias+torch.sum(X_unmapped[candidate_nids_unconnected], dim=0).view(1,-1),
            ], dim=0).view(-1)
        return X

    def _intersect_neighbors_of_set1_with_set2(self, sg_nids, candidate_nids, g):
        subgraph_and_frontier_nids = set().union(*[nx.neighbors(g, nid) for nid in candidate_nids])
        return [nid for nid in candidate_nids if len(sg_nids.intersection(subgraph_and_frontier_nids)) > 0]

class SimilarityPartitionDecoder(torch.nn.Module):
    def __init__(self, dim, partition_func_li, partition_decoder):
        super(SimilarityPartitionDecoder, self).__init__()
        self.partition_func_li = partition_func_li
        self.partition_decoder = partition_decoder
        self.dim = dim

    def forward(self, Xsgq, Xsgt, logger):
        assert Xsgq.shape[0] == Xsgt.shape[0] == self.dim*3
        sim_partition_li = []
        for i in range(3):
            Xsgq_partition, Xsgt_partition = Xsgq[i*self.dim:(i+1)*self.dim], Xsgt[i*self.dim:(i+1)*self.dim]
            sim_partition = self.partition_func_li[i](Xsgq_partition, Xsgt_partition, logger)
            sim_partition_li.append(sim_partition)
        sim = self.partition_decoder(torch.cat(sim_partition_li, dim=0).view(-1))
        return sim

class SimilaritySimpleSumDecoder(torch.nn.Module):
    def __init__(self):
        super(SimilaritySimpleSumDecoder, self).__init__()

    def forward(self, Xsgq, Xsgt, logger):
        sim = (Xsgq+Xsgt)/2
        return sim

class SimilarityConcatDecoder(torch.nn.Module):
    def __init__(self, mlp_in, mlp_out):
        super(SimilarityConcatDecoder, self).__init__()
        self.encoder = mlp_in
        self.decoder = mlp_out

    def forward(self, Xsgq, Xsgt, logger=None):
        Xsgq_latent, Xsgt_latent = self.encoder(Xsgq), self.encoder(Xsgt)
        sim = self.decoder(torch.cat([Xsgq_latent, Xsgt_latent], dim=-1))
        # logger.sim_log.append({'Xsg':(Xsgq, Xsgt),'Xsg_enc':(Xsgq_latent, Xsgt_latent),'dec_bilinear':sim})
        return sim

class SimilarityBilinearDecoder(torch.nn.Module):
    def __init__(self, mlp_in, mlp_out, d_in, d_out):
        super(SimilarityBilinearDecoder, self).__init__()
        self.encoder = mlp_in
        self.bilinear_mat = torch.nn.Parameter(torch.randn(d_in, d_in, d_out) + torch.eye(d_in).view(d_in, d_in, 1))
        self.decoder = mlp_out

    def forward(self, Xsgq, Xsgt, logger=None):
        Xsgq_latent, Xsgt_latent = self.encoder(Xsgq), self.encoder(Xsgt)
        sim_latent = torch.einsum('ik,klj,hl->ihj', Xsgq_latent, self.bilinear_mat, Xsgt_latent)
        sim = self.decoder(sim_latent)
        return sim, sim_latent.squeeze(dim=0)

class TransformerAttention(torch.nn.Module):
    def __init__(self, mlp_query, mlp_key, mlp_value, temp):
        super(TransformerAttention, self).__init__()
        self.mlp_query = mlp_query
        self.mlp_key = mlp_key
        self.mlp_value = mlp_value
        self.temp = temp

    def forward(self, query, key, value):
        query = query + self.mlp_query(query)
        key2 = self.mlp_key(key)
        key = key + key2
        value = value + self.mlp_value(value)
        att = \
            (
                torch.matmul(query, key.transpose(0,1)) / (self.temp*np.sqrt(key.shape[1]))
            ).view(-1,1)
        out = torch.sum(torch.softmax(att, dim=0) * value, dim=0)
        # out = torch.sum(att * value, dim=0)
        return out
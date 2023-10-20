from NSUBS.src.utils import OurTimer
import torch
import torch.nn.functional as F

from NSUBS.model.OurSGM.config import FLAGS

class DVN(torch.nn.Module):
    def __init__(self, pre_encoder, encoder, decoder_policy, decoder_value, norm_li):
        super(DVN, self).__init__()
        self.pre_encoder = pre_encoder
        self.encoder = encoder
        self.decoder_policy = decoder_policy
        self.decoder_value = decoder_value
        self.norm_li = norm_li

    def encoder_wrapper(self, Xq, edge_indexq, Xt, edge_indext, gq, gt,
                        nn_map, cs_map, candidate_map, u2v_li, node_mask,
                        cache_target_embeddings):
        timer = None
        if FLAGS.time_analysis:
            timer = OurTimer()


        norm_q, norm_t = None, None #self.create_norm_vec(gq, gt)

        if FLAGS.time_analysis:
            timer.time_and_clear(f'create_norm_vec')

        Xq, Xt = \
            self.pre_encoder(Xq, Xt, nn_map)

        # Xq = self.norm_li[0](Xq)
        # Xt = self.norm_li[1](Xt)

        if FLAGS.time_analysis:
            timer.time_and_clear(f'pre_encoder')
        Xq, Xt = \
            self.encoder(
                Xq, edge_indexq, Xt, edge_indext,
                nn_map, cs_map, candidate_map,
                norm_q, norm_t, u2v_li, node_mask,
                cache_target_embeddings
            )

        if FLAGS.time_analysis:
            timer.time_and_clear(f'encoder')
            # timer.print_durations_log()

        if FLAGS.apply_norm:
            Xq = self.norm_li[2](Xq)
            Xt = self.norm_li[3](Xt)

        if FLAGS.time_analysis:
            timer.time_and_clear(f'apply_norm')
            timer.print_durations_log()
        return Xq, Xt

    def forward(self, Xq, edge_indexq, Xt, edge_indext, gq, gt,
                nn_map, cs_map, candidate_map, u2v_li, node_mask,
                cache_embeddings, execute_action, query_tree,
                u=None, v_li=None):

        timer = None
        if FLAGS.time_analysis:
            timer = OurTimer()


        Xq, Xt = \
            self.encoder_wrapper(
                Xq, edge_indexq, Xt, edge_indext, gq, gt,
                nn_map, cs_map, candidate_map, u2v_li,
                node_mask, cache_embeddings,
            )

        if FLAGS.time_analysis:
            timer.time_and_clear(f'encoder')

        out_value, g_emb = self.decoder_value(Xq, Xt, gq, gt, nn_map, candidate_map, node_mask, None)

        if FLAGS.time_analysis:
            timer.time_and_clear(f'decoder value')

        out_policy, bilin_emb = self.decoder_policy(Xq, Xt, u, v_li, gq, gt,\
                                                     nn_map, cs_map, candidate_map, execute_action, query_tree, g_emb)

        if FLAGS.time_analysis:
            timer.time_and_clear(f'decoder policy')
        out_policy = out_policy.view(-1)
        out_other = {
            'Xq':Xq,
            'Xt':Xt,
            'g_emb':g_emb.detach().cpu().numpy(),
            'bilin_emb':None
        }
        # {
        #     'Xq':Xq,
        #     'Xt':Xt,
        #     'g_emb':g_emb.detach().cpu().numpy(),
        #     'bilin_emb':bilin_emb.detach().cpu().numpy(),
        # }

        if FLAGS.time_analysis:
            timer.print_durations_log()
        return out_policy, out_value, out_other

    def create_norm_vec(self, gq, gt):
        norm_q = torch.tensor([gq.degree(nid) + 1e-8 for nid in range(gq.number_of_nodes())], dtype=torch.float32, device=FLAGS.device).view(-1,1)
        norm_t = torch.tensor([gt.degree(nid) + 1e-8 for nid in range(gt.number_of_nodes())], dtype=torch.float32, device=FLAGS.device).view(-1,1)
        return norm_q, norm_t
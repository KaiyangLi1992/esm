from config import FLAGS, EXP_DEPTH
from saver import saver
from data_loader import get_data_loader_wrapper
from train import train
from test import test
from model_glsearch import GLS
from utils_our import load_replace_flags
from utils import OurTimer, save_pickle
from dvn_wrapper import create_dvn

import time
import torch
import traceback
import random, numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import defaultdict
from os.path import join

def get_d_in_raw():
    if FLAGS.do_train:
        dl = get_data_loader_wrapper('train')
    elif FLAGS.do_test:
        dl = get_data_loader_wrapper('test')
    else:
        assert False
    gp = next(dl)
    return gp.get_d_in_raw()

def _create_model(d_in_raw):
    if FLAGS.matching_order == 'nn':
        if FLAGS.load_model != 'None':
            load_replace_flags(FLAGS.load_model)
            saver.log_new_FLAGS_to_model_info()
            if FLAGS.glsearch:
                model = GLS() # create here since FLAGS have been updated_create_model
            else:
                model = create_dvn(d_in_raw, FLAGS.d_enc)
                # model = DGMC() # create here since FLAGS have been updated
            ld = torch.load(FLAGS.load_model, map_location=FLAGS.device)
            model.load_state_dict(ld)
            saver.log_info(f'Model loaded from {FLAGS.load_model}')
        else:
            if FLAGS.glsearch:
                model = GLS()
            else:
                model = create_dvn(d_in_raw, FLAGS.d_enc)
                # model = DGMC()
        saver.log_model_architecture(model, 'model')
        return model.to(FLAGS.device)
    else:
        return None

model = _create_model(get_d_in_raw())
train_loader = get_data_loader_wrapper('train')
for i, gp in enumerate(train_loader):
    gq, gt, CS, daf_path_weights, true_nn_map = gp.gq, gp.gt, gp.CS, gp.daf_path_weights, gp.true_nn_map

    nn_map = {}
    cs_map = CS[0]
    u = 0

    v_li = cs_map[u] 
    candidate_map = {u:v_li }
    # gq.x = gq.x.cuda() 
    # gt.x = gt.x.cuda()

    out_policy, out_value, out_other = \
        model(
            gq, gt, u, v_li,
            nn_map, cs_map, candidate_map,
            True,
            graph_filter=None, filter_key=None,
        )
    
import networkx as nx
# from config import FLAGS
#########################################################################
# Double Dictionary
#########################################################################
class DoubleDict():
    def __init__(self):
        self.l2r = {}
        self.r2l = {}

    def __len__(self):
        return len(self.l2r)

    def add_lr(self, l, r):
        if l not in self.l2r:
            self.l2r[l] = set()
        if r not in self.r2l:
            self.r2l[r] = set()
        self.l2r[l].add(r)
        self.r2l[r].add(l)

    def get_l2r(self, l):
        if l not in self.l2r:
            return set()
        else:
            return self.l2r[l]

    def get_r2l(self, r):
        if r not in self.r2l:
            return set()
        else:
            return self.r2l[r]


#########################################################################
# Action Edge
#########################################################################
class ActionEdge(object):
    def __init__(self, action, reward,
                 pruned_actions=None, exhausted_v=None, exhausted_w=None):
        self.action = action
        self.reward = reward

        # input to DQN
        self.pruned_actions = DoubleDict() \
            if pruned_actions is None else deepcopy(pruned_actions)
        self.exhausted_v = set() \
            if exhausted_v is None else deepcopy(exhausted_v)
        self.exhausted_w = set() \
            if exhausted_w is None else deepcopy(exhausted_w)

        # for search tree
        self.state_prev = None
        self.state_next = None

    def link_action2state(self, cur_state, next_state):
        self.state_prev = cur_state
        self.state_next = next_state
        cur_state.action_next_list.append(self)
        next_state.action_prev = self


class SearchTree(object):
    def __init__(self, root):
        self.root = root
        self.nodes = {root}
        self.edges = set()
        self.nxgraph = nx.Graph()

        # add root to nxgraph
        self.cur_nid = 0
        root.nid = self.cur_nid
        self.nxgraph.add_node(self.cur_nid)
        self.nid2ActionEdge = {}
        self.cur_nid += 1

        node_stack = [root]
        while len(node_stack) > 0:
            node = node_stack.pop()
            for action_edge in node.action_next_list:
                node_next = action_edge.state_next
                node_next.cur_nid = self.cur_nid
                self.cur_nid += 1
                self.nodes.add(node_next)
                assert action_edge not in self.edges
                self.edges.add(action_edge)
                self.nxgraph.add_node(node_next.cur_nid)
                self.nid2ActionEdge[node_next.cur_nid] = action_edge
                self.nxgraph.add_edge(node.nid, node_next.cur_nid)
                node_stack.append(node_next)

    def link_states(self, cur_state, action_edge, next_state, q_pred, discount):
        action_edge.link_action2state(cur_state, next_state)

        assert cur_state in self.nodes
        self.nodes.add(next_state)
        assert action_edge not in self.edges
        self.edges.add(action_edge)

        # add edge, node to nxgraph
        assert cur_state.nid is not None
        next_state.nid = self.cur_nid
        self.nxgraph.add_node(self.cur_nid)
        self.nid2ActionEdge[self.cur_nid] = action_edge
        self.nxgraph.add_edge(cur_state.nid, self.cur_nid)
        eid = (cur_state.nid, self.cur_nid)
        self.assign_val_to_edge(eid, 'q_pred', q_pred)

        # self.assign_val_to_edge(eid, 'q_pred', q_pred.item())

        # accumulate the reward
        next_state.cum_reward = \
            self.get_next_cum_reward(cur_state, action_edge, discount)
        self.cur_nid += 1

    def get_next_cum_reward(self, cur_state, action_edge, discount):
        next_cum_reward = \
            cur_state.cum_reward + (discount ** cur_state.num_steps) * action_edge.reward
        return next_cum_reward

    def assign_val_to_node(self, nid, key, val):
        self.nxgraph.nodes[nid][key] = val

    def assign_val_to_edge(self, eid, key, val):
        self.nxgraph.edges[eid][key] = val
    def assign_v_search_tree(self, discount):  # , g1, g2):
        self.root.assign_v(discount)

def get_natts_hash(node):
    if 'fuzzy_matching' in FLAGS.reward_calculator_mode:
        natts = []
    else:
        natts = FLAGS.node_feats_for_sm
    natts_hash = tuple([node[natt] for natt in natts])
    return natts_hash


def unroll_bidomains(natts2bds):
    bidomains = [bd for bds in natts2bds.values() for bd in bds]
    return bidomains
class Bidomain(object):
    def __init__(self, left, right, natts, bid=None):
        self.left = left
        self.right = right
        self.natts = natts
        self.bid = bid

    def __len__(self):
        return len(self.left) * len(self.right)
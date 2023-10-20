class QueryNode():
    def __init__(self, nid, level, parents):
        self.nid = nid
        self.level = level
        self.parents = parents
        self.children = []
    def get_parent_nids(self):
        return [parent.nid for parent in self.parents]
    def get_one_level_lower_children(self):
        return [child for child in self.children if child.level == self.level+1]

class QueryTree():
    def __init__(self, g_query, levels2u2v):
        assert 0 in levels2u2v
        assert 0 == min(levels2u2v.keys())
        for level, u2v in sorted(levels2u2v.items()):
            if level == 0:
                u_root, = u2v.keys()
                self.root = QueryNode(u_root, 0, [])
                self.nid2node = {u_root: self.root}
            else:
                u_li = u2v.keys()
                for u in u_li:
                    nid_li_parents = set.intersection(set(g_query.neighbors(u)), set(self.nid2node.keys()))
                    self.nid2node[u] = \
                        QueryNode(u, level, [self.nid2node[nid] for nid in nid_li_parents])
                    for u_parent in nid_li_parents:
                        self.nid2node[u_parent].children.append(self.nid2node[u])

from collections import defaultdict
def get_levels2u2v(CS):
    u2v, u2levels = CS
    levels2u2v = defaultdict(dict)
    for u,level in u2levels.items():
        levels2u2v[level][u] = u2v[u]
    return levels2u2v
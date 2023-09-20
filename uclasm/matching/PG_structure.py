from itertools import product
import random
import networkx as nx
def swap_source_target(graph):
    new_graph = nx.DiGraph()
    
    for source, target in graph.edges():
        new_graph.add_edge(target, source)  # 交换源节点和目标节点
        
    return new_graph

class State(object):
    def __init__(self,g1,g2,g1_reverse=None,g2_reverse=None,exhausted_u=set(),exhausted_v=set(),nn_mapping={}):
        self.g1 = g1
        self.g2 = g2
        if g1_reverse == None:
            self.g1_reverse = swap_source_target(g1)
        else:
            self.g1_reverse = g1_reverse
        if g2_reverse == None:
            self.g2_reverse = swap_source_target(g2)
        else:
            self.g2_reverse = g2_reverse
        self.nn_mapping = nn_mapping
        self.action_space = self.get_action_space()
    def get_action_space(self):
        attr2node1 = self.g1.attr_dict.copy()
        attr2node2 = self.g2.attr_dict.copy()
        mapped_u = set(self.nn_mapping.keys())
        mapped_v = set(self.nn_mapping.values())
        for key in attr2node1:
            attr2node1[key] = attr2node1[key] - mapped_u
        for key in attr2node2:
            attr2node2[key] = attr2node2[key] - mapped_v

        result_list = []
        for key in attr2node1:
            if key in attr2node2:
                cartesian_product = product(attr2node1[key], attr2node2[key])
                result_list.extend(cartesian_product)
        while len(result_list) < 500:
            result_list.append((-1, -1))

        # 如果列表长度超过500，随机选择500个元素构成新列表
        if len(result_list) > 500:
            result_list = random.sample(result_list, 500)
        return result_list




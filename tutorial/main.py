"""Example usage of the uclasm package for finding subgraph isomorphisms."""
import sys 
sys.path.append("/home/kli16/ISM_custom/esm/uclasm/") 
sys.path.append("/home/kli16/ISM_custom/esm/")
print(sys.path)
import uclasm
from uclasm.matching import *
import numpy as np
tmplt = uclasm.load_edgelist("./tutorial/template.csv",
                             file_source_col="Source",
                             file_target_col="Target")

world = uclasm.load_edgelist("./tutorial/world.csv",
                             file_source_col="Source",
                             file_target_col="Target")
# fixed_costs_ = np.load('./fixed_cost_email.npy')
# fixed_costs = np.zeros(fixed_costs_.shape)
# for i in range(fixed_costs_.shape[0]):
#     for j in range(fixed_costs_.shape[1]):
#         if fixed_costs_[i][j] == 0:
#             fixed_costs[tmplt.node_idxs[str(i)]][world.node_idxs[str(j)]] = np.inf

# smp = uclasm.MatchingProblem(tmplt, world,ground_truth_provided=False,fixed_costs=fixed_costs,global_cost_threshold=1e6)
smp = uclasm.MatchingProblem(tmplt, world,ground_truth_provided=False,global_cost_threshold=1e6)
# local_cost_bound.nodewise(smp)
local_cost_bound.edgewise(smp)
global_cost_bound.from_local_bounds(smp)
solutions = search.greedy_best_k_matching(smp,k=1,nodewise=False, verbose=True)

# uclasm.matching.local_cost_bound.nodewise(smp)
# uclasm.matching.global_cost_bound.from_local_bounds(smp)

print(smp)


##test
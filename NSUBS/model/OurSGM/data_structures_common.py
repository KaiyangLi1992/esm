from heapq import heappush, heappop
import itertools

#########################################################################
# Search Stack
#########################################################################
'''
Modified from python documentation for PriorityQ implementation:
https://docs.python.org/3/library/heapq.html
'''
import numpy as np

class StackHeap:
    def __init__(self):
        self.pq = []  # list of entries arranged in a heap
        self.sk = []
        self.priority_map = {}
        self.entry_finder = {}  # mapping of tasks to entries
        self.REMOVED = '<removed-task>'  # placeholder for a removed task
        self.counter = itertools.count()  # unique sequence count

    def __len__(self):
        return len(self.sk)

    def add(self, task, priority=0, front=False):
        'Add a new task or update the priority of an existing task'
        if task in self.entry_finder:
            self.remove_task(task, delete=False)
        self.priority_map[task] = priority
        count = next(self.counter)
        entry = [-priority, count, task]
        self.entry_finder[task] = entry
        heappush(self.pq, entry)
        if front:
            self.sk = [entry] + self.sk
        else:
            self.sk.append(entry)

    def is_task_in_stack_heap(self, task):
        return task in self.entry_finder and self.entry_finder[task][-1] != self.REMOVED

    def remove_task(self, task, delete):
        'Mark an existing task as REMOVED.  Raise KeyError if not found.'
        if delete:
            entry = self.entry_finder[task]
            self.sk.remove(entry)
            del self.entry_finder[task]
        else:
            entry = self.entry_finder.pop(task)
            self.sk.remove(entry)
            entry[-1] = self.REMOVED
        del self.priority_map[task]

    def get_task(self, method):
        if method == 'heap':
            priority, count, task = self.pq[0]
            while task is self.REMOVED:
                heappop(self.pq)
                priority, count, task = self.pq[0]
        elif method == 'stack':
            priority, count, task = self.sk[-1]
        else:
            assert False
        return task, -priority

    def pop_task(self, method):
        if method == 'heap':
            # Remove and return the lowest priority task.
            # Raise KeyError if empty.
            while self.pq:
                priority, count, task = heappop(self.pq)
                if task is not self.REMOVED:
                    self.remove_task(task, delete=True)
                    return task, -priority
            raise KeyError('pop from an empty priority queue')
        elif method == 'prob_heap':
            task_priority_li =\
                [[task, self.entry_finder[task][0]] for task in self.entry_finder if self.entry_finder[task][-1] != self.REMOVED]
            assert len(task_priority_li) > 0
            # priorities need to be inversed and made positive bc min heap
            priority_prob = -np.array([priority for _, priority in task_priority_li]) + 1e-8
            priority_prob /= np.sum(priority_prob)
            task_priority_li_idx, = np.random.choice(len(task_priority_li), 1, p=priority_prob)
            task, priority = task_priority_li[task_priority_li_idx]
            self.remove_task(task, delete=False)
        elif method == 'stack':
            priority, count, task = self.sk[-1]
            self.remove_task(task, delete=False)
        else:
            assert False

        return task, -priority


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
# Replay Buffer Entry
#########################################################################
class QsaObj():
    def __init__(self, u, v_li, v_li_sampled, r_accum, v_pi_logits_li, v_r_accum_li_sampled,
                 nn_map, filter_key, gq, gt, CS, candidate_map, graph_filter):
        self.u = u
        self.v_li = v_li
        self.v_li_sampled = v_li_sampled
        self.r_accum = r_accum
        self.v_pi_logits_li = v_pi_logits_li
        self.v_r_accum_li_sampled = v_r_accum_li_sampled
        self.nn_map = nn_map
        self.gq = gq
        self.gt = gt
        self.CS = CS
        self.candidate_map = candidate_map
        self.graph_filter = graph_filter
        self.filter_key = filter_key


#########################################################################
# GlobalSearchParams
#########################################################################
class GlobalSearchParams:
    def __init__(self, gq, gt, CS, daf_path_weights, model, graph_filter,
                 override_use_heuristic, eps, logger, gid, MCTS, align_matrix):
        self.gq = gq
        self.gt = gt
        self.CS = CS
        self.daf_path_weights = daf_path_weights
        self.model = model
        self.graph_filter = graph_filter
        self.override_use_heuristic = override_use_heuristic
        self.eps = eps
        self.logger = logger
        self.gid = gid
        self.MCTS = MCTS
        self.align_matrix = align_matrix

    def get_params(self):
        return self.gq, self.gt, self.CS, self.daf_path_weights,\
               self.model, self.graph_filter, self.override_use_heuristic, \
               self.eps, self.align_matrix

import numpy as np
from torch.utils.data.sampler import Sampler
import copy
import itertools
from collections import defaultdict
from typing import Optional, List
from reid.utils import comm

def no_index(a, b):
    assert isinstance(a, list)
    return [i for i, j in enumerate(a) if j != b]

def reorder_index(batch_indices, world_size):
    mini_batchsize = len(batch_indices) // world_size
    reorder_indices = []
    for i in range(0, mini_batchsize):
        for j in range(0, world_size):
            reorder_indices.append(batch_indices[i + j * mini_batchsize])
    return reorder_indices

class BalancedIdentitySampler(Sampler):
    def __init__(self, data_source: List, mini_batch_size: int, num_instances: int, seed: Optional[int] = None):
        self.data_source = data_source
        self.num_instances = num_instances
        self.num_pids_per_batch = mini_batch_size // self.num_instances

        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()
        self.batch_size = mini_batch_size * self._world_size

        self.index_pid = dict()
        self.pid_cam = defaultdict(list)
        self.pid_index = defaultdict(list)

        for index, info in enumerate(data_source):
            pid = info[1]
            camid = info[2]
            self.index_pid[index] = pid
            self.pid_cam[pid].append(camid)
            self.pid_index[pid].append(index)

        self.pids = sorted(list(self.pid_index.keys()))
        self.num_identities = len(self.pids)     
        if seed is None: 
            seed = comm.shared_random_seed()
        self._seed = int(seed)

        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()

    def __iter__(self): 
        start = self._rank     
        yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)

    def _infinite_indices(self): 
        np.random.seed(self._seed)
        while True:
           
            identities = np.random.permutation(self.num_identities)
            drop_indices = self.num_identities % (self.num_pids_per_batch * self._world_size)
            if drop_indices: identities = identities[:-drop_indices]

            batch_indices = []
            for kid in identities:
                i = np.random.choice(self.pid_index[self.pids[kid]])
                _, i_pid, i_cam = self.data_source[i]
                batch_indices.append(i)
                pid_i = self.index_pid[i]
                cams = self.pid_cam[pid_i]
                index = self.pid_index[pid_i]
                select_cams = no_index(cams, i_cam)

                if select_cams:
                    if len(select_cams) >= self.num_instances:
                        cam_indexes = np.random.choice(select_cams, size=self.num_instances - 1, replace=False)
                    else:
                        cam_indexes = np.random.choice(select_cams, size=self.num_instances - 1, replace=True)
                    for kk in cam_indexes:
                        batch_indices.append(index[kk])
                else:
                    select_indexes = no_index(index, i)
                    if not select_indexes:                    
                        ind_indexes = [0] * (self.num_instances - 1)
                    elif len(select_indexes) >= self.num_instances:
                        ind_indexes = np.random.choice(select_indexes, size=self.num_instances - 1, replace=False)
                    else:
                        ind_indexes = np.random.choice(select_indexes, size=self.num_instances - 1, replace=True)

                    for kk in ind_indexes:
                        batch_indices.append(index[kk])

                if len(batch_indices) == self.batch_size:
                    yield from reorder_index(batch_indices, self._world_size)
                    batch_indices = []


class NaiveIdentitySampler(Sampler):
    def __init__(self, data_source: str, mini_batch_size: int, num_instances: int, seed: Optional[int] = None):
       
        self.data_source = data_source      
        self.num_instances = num_instances       
        self.num_pids_per_batch = mini_batch_size // self.num_instances
        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()
        self.batch_size = mini_batch_size * self._world_size      
        self.pid_index = defaultdict(list)

        for index, info in enumerate(data_source):
            pid = info[1]
            self.pid_index[pid].append(index)       
        self.pids = sorted(list(self.pid_index.keys()))       
        self.num_identities = len(self.pids)        
        if seed is None:
            seed = comm.shared_random_seed()
        self._seed = int(seed)

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)
    def _infinite_indices(self):
        np.random.seed(self._seed)
        while True:        
            avl_pids = copy.deepcopy(self.pids)            
            batch_idxs_dict = {}            
            batch_indices = []            
            while len(avl_pids) >= self.num_pids_per_batch:                
                selected_pids = np.random.choice(avl_pids, self.num_pids_per_batch, replace=False).tolist()               
                for pid in selected_pids:                    
                    if pid not in batch_idxs_dict:                        
                        idxs = copy.deepcopy(self.pid_index[pid])                        
                        if len(idxs) < self.num_instances:
                            idxs = np.random.choice(idxs, size=self.num_instances, replace=True).tolist()                       
                        np.random.shuffle(idxs)                       
                        batch_idxs_dict[pid] = idxs                  
                    avl_idxs = batch_idxs_dict[pid]
                    for _ in range(self.num_instances):
                        batch_indices.append(avl_idxs.pop(0))                   
                    if len(avl_idxs) < self.num_instances: avl_pids.remove(pid)            
                if len(batch_indices) == self.batch_size:
                    yield from reorder_index(batch_indices, self._world_size)
                    batch_indices = []

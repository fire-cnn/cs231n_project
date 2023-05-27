import torch
import numpy as np
from torch.utils.data import Sampler

class BalancedSampler(Sampler):
    def __init__(self, dataset, batch_size):

        self.indices = dataset.Y.index
        weights = ( 1 / dataset.Y.value_counts(sort=False))
        weights = weights.to_dict()
        weights = [weights[i] for i in dataset.Y]
        self.weights = torch.tensor(weights, dtype = torch.double)

    def __len__(self):
        return self.indice // self.batch_size

    def __iter__(self):
        batches = []
        for _ in range(len(self)):
            batch = [self.indices[i]  for i in 
                     torch.multinomial(self.weights, self.num_samples, replacement=True)]
            batches.append(batch)

        return iter(batches)


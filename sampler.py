import torch
from torch.utils.data import Dataset, Sampler
import numpy as np
from typing import Iterator, Union


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset_df, sampling_factor: Union[float, str] = None):
        self.df = dataset_df
        self.filenames = self.df[0].unique()
        self.length = len(self.filenames)
        self.dataset = dict()
        self.balanced_max = 0
        self.balanced_min = 99999
        # Save all the indices for all the classes
        for idx in range(0, self.length):
            label = self._get_label(idx)
            if label not in self.dataset:
                self.dataset[label] = list()
            self.dataset[label].append(idx)
            if len(self.dataset[label]) > self.balanced_max:
                self.balanced_max = len(self.dataset[label])
            if len(self.dataset[label]) < self.balanced_min:
                self.balanced_min = len(self.dataset[label])

        if sampling_factor is None:
            raise Exception("Sampling factor undecided")

        if sampling_factor == 'oversampling' or (0 <= sampling_factor <= 1.0):
            # Oversample the classes to the amount of the linear interpolate size between the smallest and the largest
            if sampling_factor == 'oversampling':
                sampling_factor = 1.0
            interclass_dist = self.balanced_max - self.balanced_min
            required_num = self.balanced_min + int(sampling_factor * interclass_dist)

        elif sampling_factor == 'undersampling' or (-1.0 < sampling_factor < 0):
            # Undersample the classes to the amount of the largest class by a factor
            if sampling_factor == 'undersampling':
                sampling_factor = -1.0
            required_num = int(self.balanced_max * -sampling_factor)

        for label in self.dataset:
            diff = required_num - len(self.dataset[label])
            if diff > 0:
                self.dataset[label].extend(
                    np.random.choice(self.dataset[label], size=diff)
                )
            else:
                self.dataset[label].remove(
                    np.random.choice(self.dataset[label], size=-diff, replace=False)
                )
        self.keys = list(self.dataset.keys())
        self.currentkey = 0
        self.indices = [-1] * len(self.keys)


    def __iter__(self) -> Iterator[int]:
        while self.indices[self.currentkey] < self.balanced_max - 1:
            self.indices[self.currentkey] += 1
            yield self.dataset[self.keys[self.currentkey]][self.indices[self.currentkey]]
            self.currentkey = (self.currentkey + 1) % len(self.keys)
        self.indices = [-1] * len(self.keys)

    def _get_label(self, idx):
        file_name = getFileNameFromDf(self.df, idx)
        label = getLabelFromFilename(file_name)
        return label

    def __len__(self):
        return self.balanced_max * len(self.keys)

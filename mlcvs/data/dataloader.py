#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
PyTorch Lightning DataModule object for DictionaryDatasets.
"""

__all__ = ["FastDictionaryLoader"]


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

from typing import Union
import torch
from torch.utils.data import Subset
from mlcvs.data import DictionaryDataset
from mlcvs.core.transform.utils import Statistics


# =============================================================================
# FAST DICTIONARY LOADER CLASS
# =============================================================================

class FastDictionaryLoader:
    """PyTorch DataLoader for :class:`~mlcvs.data.dataset.DictionaryDataset`s.
    
    It is much faster than ``TensorDataset`` + ``DataLoader`` because ``DataLoader``
    grabs individual indices of the dataset and calls cat (slow).

    Notes
    -----

    Adapted from https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6.

    Examples
    --------

    >>> x = torch.arange(1,11)

    A ``FastDictionaryLoader`` can be initialize from a ``dict``, a :class:`~mlcvs.data.dataset.DictionaryDataset`,
    or a ``Subset`` wrapping a :class:`~mlcvs.data.dataset.DictionaryDataset`.

    >>> # Initialize from a dictionary.
    >>> d = {'data': x.unsqueeze(1), 'labels': x**2}
    >>> dataloader = FastDictionaryLoader(d, batch_size=1, shuffle=False)
    >>> dataloader.dataset_len  # number of samples
    10
    >>> next(iter(dataloader))  # first batch
    {'data': tensor([[1]]), 'labels': tensor([1])}

    >>> # Initialize from a DictionaryDataset.
    >>> dict_dataset = DictionaryDataset(d)
    >>> dataloader = FastDictionaryLoader(dict_dataset, batch_size=2, shuffle=False)
    >>> len(dataloader)  # Number of batches
    5
    >>> batch = next(iter(dataloader))  # first batch
    >>> batch['data']
    tensor([[1],
            [2]])

    >>> # Initialize from a Subset.
    >>> train, _ = torch.utils.data.random_split(dict_dataset, [0.5, 0.5])
    >>> dataloader = FastDictionaryLoader(train, batch_size=1, shuffle=False)

    """
    def __init__(
            self,
            dataset: Union[dict, DictionaryDataset, Subset],
            batch_size: int = 0,
            shuffle: bool = True,
    ):
        """Initialize a ``FastDictionaryLoader``.

        Parameters
        ----------
        dataset : dict or DictionaryDataset or Subset or list-like.
            The dataset.
        batch_size : int, optional
            Batch size, by default 0 (==single batch).
        shuffle : bool, optional
            If ``True``, shuffle the data *in-place* whenever an
            iterator is created out of this object, by default ``True``.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    @property
    def dataset(self):
        """DictionaryDataset or list-like of DictionaryDataset: The dictionary dataset(s)."""
        return self._dataset

    @dataset.setter
    def dataset(self, dataset):
        self._dataset = self._to_dict_dataset(dataset)

    @property
    def dataset_len(self):
        """int: Number of samples in the dataset(s)."""
        return len(self.dataset)

    @property
    def batch_size(self):
        """int: Batch size."""
        return self._batch_size if self._batch_size > 0 else self.dataset_len

    @batch_size.setter
    def batch_size(self, batch_size):
        self._batch_size = batch_size

    def __iter__(self):
        if self.shuffle:
            self.indices = torch.randperm(self.dataset_len)
        else:
            self.indices = None
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        
        if self.indices is not None:
            indices = self.indices[self.i:self.i+self.batch_size]
            batch = self.dataset[indices]
        else:
            batch = self.dataset[self.i:self.i+self.batch_size]

        self.i += self.batch_size
        return batch

    def __len__(self):
        # Number of batches.
        return (self.dataset_len + self.batch_size - 1) // self.batch_size

    @property
    def keys(self):
        return self.dataset.keys
    
    def __repr__(self) -> str:
        string = f'FastDictionaryLoader(length={self.dataset_len}, batch_size={self.batch_size}, shuffle={self.shuffle})'
        return string

    def get_stats(self):
        """Compute statistics ``('mean','std','min','max')`` of the dataloader.

        Returns
        -------
        stats 
            dictionary of dictionaries with statistics
        """
        stats = {}
        for batch in iter(self):
            for k in self.keys:
                #initialize
                if k not in stats:
                    stats[k] = Statistics(batch[k])
                # or accumulate
                else:
                    stats[k].update(batch[k])

        # convert to dictionaries
        for k in stats.keys():
            stats[k] = stats[k].to_dict()

        return stats

    def _to_dict_dataset(self, d):
        """Convert Dict[Tensor] and Subset[DictionaryDataset] to DictionaryDataset."""
        # Convert to DictionaryDataset if a dict is given.
        if isinstance(d, dict):
            d = DictionaryDataset(d)
        elif isinstance(d, Subset) and isinstance(d.dataset, DictionaryDataset):
            # Retrieve selection if it a subset.
            # TODO: This is not safe for classes that inherit from Subset or DictionaryDatset.
            d = d.dataset.__class__(d.dataset[d.indices])
        return d


if __name__ == '__main__':
    import doctest
    doctest.testmod()

from abc import ABC, abstractmethod
from typing import Tuple, List, Union
import numpy as np
import random, logging

Dataset = Tuple[np.ndarray, np.ndarray]

log = logging.getLogger(__name__)


class DataSplitter(ABC):
    
    @abstractmethod
    def split(self, data: Dataset, num_splits: int) -> List[Dataset]:
        pass
    
def create_from_contiguous_shards(dataset: Dataset, num_clients: int, shard_size: int):
    train_img, train_label = dataset

    shard_start_index = [i for i in range(0, len(train_img), shard_size)]
    random.shuffle(shard_start_index)
    
    num_shards = len(shard_start_index) // num_clients
    local_datasets = []
    for client_id in range(num_clients):
        _index = num_shards * client_id
        img = np.concatenate([train_img[shard_start_index[_index + i]:
                                        shard_start_index[_index + i] + shard_size] 
                              for i in range(num_shards)], axis=0)

        label = np.concatenate([train_label[shard_start_index[_index + i]:
                                            shard_start_index[_index + i] + shard_size] 
                                for i in range(num_shards)], axis=0)
        local_datasets.append((img, label))
        
    return local_datasets

def create_non_iid(dataset: Dataset, num_clients: int, shard_size: int) -> List[Dataset]:
    train_img, train_label = dataset
    train_sorted_index = np.argsort(train_label)
    train_img = train_img[train_sorted_index]
    train_label = train_label[train_sorted_index]
    
    return create_from_contiguous_shards((train_img, train_label), num_clients, shard_size)

def partition_class_samples_with_dirichlet_distribution(N, alpha, client_num, idx_batch, idx_k):
    np.random.shuffle(idx_k)
    # using dirichlet distribution to determine the unbalanced proportion for each client (client_num in total)
    # e.g., when client_num = 4, proportions = [0.29543505 0.38414498 0.31998781 0.00043216], sum(proportions) = 1
    proportions = np.random.dirichlet(np.repeat(alpha, client_num))

    # get the index in idx_k according to the dirichlet distribution
    proportions = np.array([p * (len(idx_j) < N / client_num) for p, idx_j in zip(proportions, idx_batch)])
    proportions = proportions / proportions.sum()
    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

    # generate the batch list for each client
    idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
    std_size = np.std([len(idx_j) for idx_j in idx_batch])

    return idx_batch, std_size

def non_iid_partition_with_dirichlet_distribution(label_list: np.ndarray,
                                                  client_num: int,
                                                  classes: int,
                                                  alpha: float,
                                                  max_iter: int):
    """
        Obtain sample index list for each client from the Dirichlet distribution.
        This LDA method is first proposed by :
        Measuring the Effects of Non-Identical Data Distribution for
        Federated Visual Classification (https://arxiv.org/pdf/1909.06335.pdf).
        This can generate nonIIDness with unbalance sample number in each label.
        The Dirichlet distribution is a density over a K dimensional vector p whose K components are positive and sum to 1.
        Dirichlet can support the probabilities of a K-way categorical event.
        In FL, we can view K clients' sample number obeys the Dirichlet distribution.
        For more details of the Dirichlet distribution, please check https://en.wikipedia.org/wiki/Dirichlet_distribution
        Parameters
        ----------
            label_list : the label list from classification/segmentation dataset
            client_num : number of clients
            classes: the number of classification (e.g., 10 for CIFAR-10) OR a list of segmentation categories
            alpha: a concentration parameter controlling the identicalness among clients.
            task: CV specific task eg. classification, segmentation
        Returns
        -------
            samples : ndarray,
                The drawn samples, of shape ``(size, k)``.
    """


    net_dataidx_map = {}
    K = classes

    # For multiclass labels, the list is ragged and not a numpy array
    N = len(label_list)

    # guarantee the minimum number of sample in each client
    iter_counter = 0

    best_std = np.inf
    best_idx_batch = [[] for _ in range(client_num)]

    while iter_counter < max_iter:
        iter_counter += 1
        idx_batch = [[] for _ in range(client_num)]

        # for each classification in the dataset
        for k in range(K):
            # get a list of batch indexes which are belong to label k
            idx_k = np.where(label_list == k)[0]
            idx_batch, std_size = partition_class_samples_with_dirichlet_distribution(N, alpha, client_num, idx_batch,
                                                                                      idx_k)

        if std_size < best_std:
            best_std = std_size
            best_idx_batch = idx_batch.copy()
            log.info(f'Partitioning with Dirichlet: best std: {std_size}, iteration number: {iter_counter}')

    for i in range(client_num):
        np.random.shuffle(best_idx_batch[i])
        net_dataidx_map[i] = best_idx_batch[i]

    return net_dataidx_map



def create_using_dirichlet_distr(dataset: Dataset, num_clients: int, shard_size: int, dataset_num_classes: int,
                                 alpha: float, max_iter: int, rebalance: bool):
    samples, targets = dataset
    d = non_iid_partition_with_dirichlet_distribution(
        targets, num_clients, dataset_num_classes, alpha, max_iter)

    if rebalance:
        storage = []
        for i in range(len(d)):
            if len(d[i]) > (shard_size):
                difference = round(len(d[i]) - (shard_size))
                toSwitch = np.random.choice(
                    d[i], difference, replace=False).tolist()
                storage += toSwitch
                d[i] = list(set(d[i]) - set(toSwitch))

        for i in range(len(d)):
            if len(d[i]) < (shard_size):
                difference = round((shard_size) - len(d[i]))
                toSwitch = np.random.choice(
                    storage, difference, replace=False).tolist()
                d[i] += toSwitch
                storage = list(set(storage) - set(toSwitch))

        for i in range(len(d)):
            if len(d[i]) != (shard_size):
                log.warning(f'There are some clients with more than {shard_size} images')

    # Lista contenente per ogni client un'istanza di Cifar10LocalDataset ->local_datasets[client]
    local_datasets = []
    for client_id in d.keys():
        # img = np.concatenate( [train_img[list_indexes_per_client_subset[client_id][c]] for c in range(n_classes)],axis=0)
        img = samples[d[client_id]]
        # label = np.concatenate( [train_label[list_indexes_per_client_subset[client_id][c]] for c in range(n_classes)],axis=0)
        label = targets[d[client_id]]
        local_datasets.append((img, label))

    return local_datasets


class DirichletDataSplitter(DataSplitter):
    def __init__(self, alpha: Union[int, float], rebalance: bool = True, max_iter_rebalance: int = 100):
        assert alpha >= 0, f"Concentration parameter of Dirichlet distribution must be >= 0, got {alpha}"
        self.alpha = alpha
        self.rebalance = rebalance
        self.max_iter_rebalance = max_iter_rebalance
        

    def split(self, data: Dataset, num_splits: int) -> List[Dataset]:
        split_size = len(data[0]) // num_splits
        log.info(f"Splitting the dataset into {num_splits} shards of size {split_size}, with alpha={self.alpha}")
        if self.alpha == 0:
            return create_non_iid(data, num_splits, split_size)
        else:
            num_classes = np.max(data[1]) + 1
            return create_using_dirichlet_distr(data, num_splits, split_size, num_classes, self.alpha,
                                                self.max_iter_rebalance, self.rebalance)
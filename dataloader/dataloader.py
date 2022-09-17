import torch

from configs.dataset_config import cfg as dataset_cfg
from configs.train_config import cfg as train_cfg
from datasets.cifar10 import Cifar10
from datasets.dataset_type import DatasetType


def get_dataloaders():

    train_dataset = Cifar10(dataset_cfg=dataset_cfg, dataset_type=DatasetType.Train)
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=train_cfg.batch_size, shuffle=True, drop_last=True)

    test_dataset = Cifar10(dataset_cfg=dataset_cfg, dataset_type=DatasetType.Test)
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=train_cfg.batch_size)

    return train_dl, test_dl

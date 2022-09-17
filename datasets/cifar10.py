from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
import numpy as np
from torchvision import transforms
from PIL import Image

from datasets.dataset_type import DatasetType


class Cifar10(Dataset):
    r"""
    https://www.cs.toronto.edu/~kriz/cifar.html
    This class is a wrapper over the default pytorch class for ease of use for the anomaly detection task.
    Parameter 'anomaly_class' is responsible for which class will be considered anomalous, while the rest are normal.
    Available classes:
                     'airplane'
                     'automobile'
                     'bird'
                     'cat'
                     'deer'
                     'dog'
                     'frog'
                     'horse'
                     'ship'
                     'truck'
    """
    def __init__(self, dataset_cfg, dataset_type: DatasetType):

        self.cfg = dataset_cfg
        self.dataset_type = dataset_type

        self.images = []
        self.labels = []
        
        if dataset_type & DatasetType.Train:
            _dataset = CIFAR10(root=str(self.cfg.data_path), train=True, download=True, transform=ToTensor())
            anomaly_class_idx = _dataset.class_to_idx[self.cfg.anomaly_class]
            imgs = _dataset.data[np.array(_dataset.targets) != anomaly_class_idx]
            lbls = np.zeros((imgs.shape[0],), dtype=int)
            self.images.append(imgs)
            self.labels.append(lbls)
            
        if dataset_type & DatasetType.Test:
            _dataset = CIFAR10(root=str(self.cfg.data_path), train=False, download=True, transform=ToTensor())
            anomaly_class_idx = _dataset.class_to_idx[self.cfg.anomaly_class]
            self.images.append(_dataset.data)
            self.labels.append((np.array(_dataset.targets) == anomaly_class_idx).astype(int))
        
        self.images = np.concatenate(self.images)
        self.labels = np.concatenate(self.labels)

    def __len__(self):
        return self.labels.shape[0]

    def apply_augmentation(self, img):
        if self.dataset_type & DatasetType.Train:
            transforms_ = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.cfg.mean,
                    std=self.cfg.std,
                )
            ])
        else:
            transforms_ = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.cfg.mean,
                    std=self.cfg.std,
                )
            ])
        return transforms_(img)

    def __getitem__(self, idx):
        image = self.apply_augmentation(Image.fromarray(self.images[idx]))
        return image, self.labels[idx]

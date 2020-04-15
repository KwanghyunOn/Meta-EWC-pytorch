import torch
from torch.utils.data.dataset import Dataset
import torchvision.datasets as dset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

import importlib


class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


class Mnist(Dataset):
    def __init__(self, root, train=True):
        self.dataset = dset.MNIST(root, train=train)

    def __getitem__(self, index):
        img, label = self.dataset[index]
        data = self.transform(img)
        return (data, label)

    def __len__(self):
        return len(self.dataset)

    def transform(self, img):
        data = TF.to_tensor(img)        # 1x28x28
        return data


class RandPermMnist(Mnist):
    def __init__(self, root, train=True):
        super().__init__(root, train)
        self.perm = torch.randperm(28*28)

    def transform(self, img):
        data = TF.to_tensor(img)
        data = torch.flatten(data)
        data = data[self.perm]
        return data


if __name__ == "__main__":
    root = "./"
    MyClass = getattr(importlib.import_module("dataset"), "RandPermMnist")
    d = MyClass(root)
    data, label = d[10]
    print(data)
    print(label)
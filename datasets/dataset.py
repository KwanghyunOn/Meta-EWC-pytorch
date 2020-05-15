import torch
from torch.utils.data.dataset import Dataset
import torchvision.datasets as dset
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class Mnist(Dataset):
    def __init__(self, root, train=True):
        self.dataset = dset.MNIST(root, train=train, download=True)

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
    def __init__(self, root, train=True, perm=None):
        super().__init__(root, train)
        if perm is None:
            self.perm = torch.randperm(28*28)
        else:
            self.perm = perm

    def transform(self, img):
        data = TF.to_tensor(img)
        data = torch.flatten(data)
        data = data[self.perm]
        return data

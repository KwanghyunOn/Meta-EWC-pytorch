import importlib
from torch.utils.data.dataset import Dataset


class DataSequenceProducer:
    def __init__(self, class_name, module_name):
        self.data_class = getattr(importlib.import_module(module_name), class_name)

    def create_list(self, len, *args, **kwargs):
        data_list = list()
        for _ in range(len):
            data_list.append(self.data_class(*args, **kwargs))
        return data_list


class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)

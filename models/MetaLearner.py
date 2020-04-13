import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class MetaLearner():
    def __init__(self, base_model, meta_model):
        self.base_model = base_model
        self.meta_model = meta_model

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.base_model.parameters(), lr = learning_rate)

    def train_base_model(self, train_dataset):
        self.base_model.train()
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        for epoch in range(num_epochs):
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = self.base_model(inputs)
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def abs_sum_of_gradient(self, dataset):
        self.base_model.train()
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
        grad_sum = torch.zeros(len(self.base_model.parameters()))

        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = self.base_model(inputs)
            loss = self.criterion(outputs, labels)
            self.optimizer.zero_grad()
            loss.backward()

            grads = []
            for param in self.base_model.parameters():
                grads.append(param.grad.view(-1))
            grads = torch.cat(grads)
            grad_sum += torch.abs(grads)

        return grad_sum

    def train_meta_learner(self, data_sequence):
        self.base_model.train()
        self.meta_model.train()

        n = len(data_sequence)
        for i in range(1, n):
            self.train_base_model(self, data_sequence[i-1])
            grad_prev = self.abs_sum_of_gradient(data_sequence[i-1])
            data_loader = DataLoader(dataset=data_sequence[i], batch_size=batch_size, shuffle=True)

            for epoch in range(num_epochs):
                for inputs, labels in data_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = self.base_model(inputs)
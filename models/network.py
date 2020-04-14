import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Network():
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def compute_gradient(self, inputs, labels):
        self.model.train()
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, labels)
        self.model.zero_grad()
        loss.backward()

        grads = []
        for param in self.model.parameters():
            grads.append(param.grad.view(-1))
        grads = torch.cat(grads)
        return grads

    def get_model_weight(self):
        weights = []
        for param in self.model.parameters():
            weights.append(param.data.view(-1))
        weights = torch.cat(weights)
        return weights

    def abs_sum_of_gradient(self, data_loader):
        self.model.train()
        grad_sum = torch.zeros(len(self.model.parameters()))

        for inputs, labels in data_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, labels)
            self.model.zero_grad()
            loss.backward()

            grads = []
            for param in self.model.parameters():
                grads.append(param.grad.view(-1))
            grads = torch.cat(grads)
            grad_sum += torch.abs(grads)

        return grad_sum

    def train_single_batch(self, inputs, labels):
        self.model.train()
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, data_loader):
        for inputs, labels in data_loader:
            self.train_single_batch(inputs, labels)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Network:
    def __init__(self, model, loss_fn, optimizer, device=None):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model.to(self.device)
        self.num_params = sum(p.numel() for p in self.model.parameters())

    def compute_gradient(self, inputs, labels):
        self.model.train()
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
        return grads

    def apply_gradient(self, grads):
        idx = 0
        for param in self.model.parameters():
            new_grad = grads[idx:(idx + param.numel())].reshape(param.grad.shape).detach()
            param.grad = new_grad
            idx += param.numel()
        return

    def get_model_weight(self):
        weights = []
        for param in self.model.parameters():
            weights.append(param.data.view(-1))
        weights = torch.cat(weights)
        return weights

    def abs_sum_of_gradient(self, data_loader):
        self.model.train()
        grad_sum = torch.zeros(self.num_params, device=self.device)

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

    def test(self, data_loader):
        self.model.eval()
        correct = total = 0
        for inputs, labels in data_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(inputs)
            pred = outputs.argmax(1)
            correct += (pred == labels).float().sum()
            total += inputs.size(0)
        return correct / total

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def compute_gradient(model, loss_fn, inputs, labels):
    outputs = model(inputs)
    loss = loss_fn(outputs, labels)
    model.zero_grad()
    loss.backward()

    grads = []
    for param in model.parameters():
        grads.append(param.grad.view(-1))
    grads = torch.cat(grads)
    return grads


def get_model_weight(model):
    weights = []
    for param in model.parameters():
        weights.append(param.data.view(-1))
    weights = torch.cat(weights)
    return weights


class MetaLearner():
    def __init__(self, main_model, meta_model):
        self.main_model = main_model
        self.meta_model = meta_model

        self.main_loss_fn = nn.CrossEntropyLoss()
        self.meta_loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.main_model.parameters(), lr = learning_rate)

    def train_main_model(self, dataset):
        self.main_model.train()
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
        for epoch in range(num_epochs):
            for inputs, labels in loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = self.main_model(inputs)
                loss = self.main_loss_fn(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def abs_sum_of_gradient(self, dataset):
        self.main_model.train()
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
        grad_sum = torch.zeros(len(self.main_model.parameters()))

        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = self.main_model(inputs)
            loss = self.main_loss_fn(outputs, labels)
            self.optimizer.zero_grad()
            loss.backward()

            grads = []
            for param in self.main_model.parameters():
                grads.append(param.grad.view(-1))
            grads = torch.cat(grads)
            grad_sum += torch.abs(grads)

        return grad_sum

    def train(self, data_sequence):
        self.main_model.train()
        self.meta_model.train()

        n = len(data_sequence)
        for i in range(1, n):
            self.train_main_model(self, data_sequence[i-1])
            prev_grads = self.abs_sum_of_gradient(data_sequence[i-1])
            joint_data_loader = DataLoader(dataset=MyDataset(data_sequence[i-1], data_sequence[i]),
                                           batch_size=batch_size,
                                           shuffle=True)
            for epoch in range(num_epochs):
               for inputs, labels in joint_data_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    prev_inputs, cur_inputs = inputs
                    prev_labels, cur_labels = labels
                    joint_inputs = torch.cat((prev_inputs, cur_inputs), dim=0)
                    joint_labels = torch.cat((prev_labels, cur_labels), dim=0)

                    cur_grads = compute_gradient(self.main_model, self.main_loss_fn, cur_inputs, cur_labels)
                    joint_grads = compute_gradient(self.main_model, self.main_loss_fn, joint_inputs, joint_labels)
                    cur_weights = get_model_weight(self.main_model)

                    train_meta_model(self.meta_model,
                                     inputs=torch.cat((prev_grads, cur_grads, cur_weights), dim=0),
                                     outputs=joint_grads)

                    train_main_model_with_meta(cur_inputs, cur_labels)
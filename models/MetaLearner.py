import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class MetaLearner():
    def __init__(self, main_net, meta_net, alpha):
        self.main_net = main_net
        self.meta_net = meta_net
        self.alpha = alpha

    def train(self, data_sequence):
        n = len(data_sequence)
        for meta_epoch in range(num_epochs_meta):
            for i in range(1, n):
                prev_data_loader = DataLoader(dataset=data_sequence[i-1], batch_size=batch_size, shuffle=True)
                self.main_net.train(prev_data_loader)

                prev_grads = self.main_net.abs_sum_of_gradient(prev_data_loader)
                prev_weights = self.main_net.get_model_weight()
                joint_data_loader = DataLoader(dataset=MyDataset(data_sequence[i-1], data_sequence[i]),
                                               batch_size=batch_size,
                                               shuffle=True)

                for main_epoch in range(num_epochs_main):
                   for inputs, labels in joint_data_loader:
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        prev_inputs, cur_inputs = inputs
                        prev_labels, cur_labels = labels
                        joint_inputs = torch.cat((prev_inputs, cur_inputs), dim=0)
                        joint_labels = torch.cat((prev_labels, cur_labels), dim=0)

                        cur_grads = self.main_net.compute_gradient(cur_inputs, cur_labels)
                        joint_grads = self.main_net.compute_gradient(joint_inputs, joint_labels)
                        cur_weights = self.main_net.get_model_weight()

                        meta_inputs = torch.cat((prev_grads, cur_grads, cur_weights), dim=0)

                        # update main network with modified gradient
                        imp = self.meta_net.model(meta_inputs)
                        cur_grads = self.alpha * imp * (cur_weights - prev_weights)
                        self.main_net.apply_gradient(cur_grads)
                        self.main_net.optimizer.step()

                        # update meta network
                        self.meta_net.train_single_batch(meta_inputs, joint_grads)

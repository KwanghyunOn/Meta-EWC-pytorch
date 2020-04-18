import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .utils import ConcatDataset


class MetaLearner:
    def __init__(self, main_net, meta_net, config, device=None):
        self.main_net = main_net
        self.meta_net = meta_net
        self.config = config
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def train(self, data_sequence):
        n = len(data_sequence)
        for meta_epoch in range(self.config.num_epochs_meta):
            for i in range(1, n):
                prev_data_loader = DataLoader(dataset=data_sequence[i-1], batch_size=self.config.batch_size,
                                              shuffle=True)
                self.main_net.train(prev_data_loader)

                prev_grads = self.main_net.abs_sum_of_gradient(prev_data_loader)
                prev_weights = self.main_net.get_model_weight()
                joint_data_loader = DataLoader(dataset=ConcatDataset(data_sequence[i-1], data_sequence[i]),
                                               batch_size=self.config.batch_size,
                                               shuffle=True)

                for main_epoch in range(self.config.num_epochs_main):
                    for prev_data, cur_data in joint_data_loader:
                        prev_inputs, prev_labels = prev_data
                        cur_inputs, cur_labels = cur_data
                        for v in [prev_inputs, cur_inputs, prev_labels, cur_labels]:
                            v = v.to(self.device)
                        joint_inputs = torch.cat((prev_inputs, cur_inputs), dim=0)
                        joint_labels = torch.cat((prev_labels, cur_labels), dim=0)

                        cur_grads = self.main_net.compute_gradient(cur_inputs, cur_labels)
                        joint_grads = self.main_net.compute_gradient(joint_inputs, joint_labels)
                        cur_weights = self.main_net.get_model_weight()

                        meta_inputs = torch.cat((prev_grads, cur_grads, cur_weights), dim=0)

                        # update main network with modified gradient
                        imp = self.meta_net.model(meta_inputs)
                        cur_grads = self.config.alpha * imp * (cur_weights - prev_weights)
                        self.main_net.apply_gradient(cur_grads)
                        self.main_net.optimizer.step()

                        # update meta network
                        self.meta_net.train_single_batch(meta_inputs, joint_grads)

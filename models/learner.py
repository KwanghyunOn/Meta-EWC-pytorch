import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .utils import ConcatDataset



class BaseLearner:
    def __init__(self, main_net, config, device=None):
        self.main_net = main_net
        self.acc_matrix = None
        self.config = config
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device


    def test(self, train_data_sequence, test_data_sequence):
        self.main_net.set_writer(self.config.writer, self.config.log_dir)
        n = len(train_data_sequence)
        self.acc_matrix = torch.zeros(n, n)
        for i in range(n):
            train_data_loader = DataLoader(dataset=train_data_sequence[i], batch_size=self.config.batch_size, shuffle=True)
            for main_epoch in range(self.config.num_epochs_per_task):
                self.main_net.train(train_data_loader)
            for j in range(i+1):
                self.acc_matrix[i][j] = self.main_net.test(DataLoader(dataset=test_data_sequence[j],
                                                                      batch_size=self.config.batch_size,
                                                                      shuffle=True))



class EWCLearner:
    def __init__(self, main_net, config, device=None):
        self.main_net = main_net
        self.acc_matrix = None
        self.config = config
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device


    def test(self, train_data_sequence, test_data_sequence):
        self.main_net.set_writer(self.config.writer, self.config.log_dir)
        n = len(train_data_sequence)
        self.acc_matrix = torch.zeros(n, n)

        self.main_net.train(DataLoader(dataset=train_data_sequence[0], batch_size=self.config.batch_size, shuffle=True))
        self.acc_matrix[0][0] = self.main_net.test(DataLoader(dataset=test_data_sequence[0],
                                                              batch_size=self.config.batch_size,
                                                              shuffle=True))
        for i in range(1, n):
            prev_data_loader = DataLoader(dataset=train_data_sequence[i-1], batch_size=self.config.batch_size,
                                          shuffle=True)
            cur_data_loader = DataLoader(dataset=train_data_sequence[i], batch_size=self.config.batch_size,
                                         shuffle=True)
            prev_grads = self.main_net.compute_avg_gradient(prev_data_loader)
            prev_weights = self.main_net.get_model_weight()
            imp = torch.abs(prev_grads)

            for main_epoch in range(self.config.num_epochs_per_task):
                for cur_inputs, cur_labels in cur_data_loader:
                    for v in [cur_inputs, cur_labels]:
                        v = v.to(self.device)
                    cur_grads = self.main_net.compute_gradient(cur_inputs, cur_labels)
                    cur_weights = self.main_net.get_model_weight()
                    cur_grads += self.config.alpha * imp * (cur_weights - prev_weights)
                    self.main_net.apply_gradient(cur_grads)
                    self.main_net.optimizer.step()
                    self.main_net.compute_loss(cur_inputs, cur_labels)

            for j in range(i+1):
                self.acc_matrix[i][j] = self.main_net.test(DataLoader(dataset=test_data_sequence[j],
                                                                      batch_size=self.config.batch_size,
                                                                      shuffle=True))



class MetaLearner:
    def __init__(self, main_net, meta_net, config, device=None):
        self.meta_net = meta_net
        self.main_net = main_net
        self.acc_matrix = None
        self.config = config
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device


    def train(self, data_sequence):
        self.meta_net.set_writer(self.config.meta_train_writer, self.config.log_dir)
        self.main_net.set_writer(self.config.main_train_writer, self.config.log_dir)
        n = len(data_sequence)

        self.main_net.train(DataLoader(dataset=data_sequence[0], batch_size=self.config.batch_size, shuffle=True))
        for meta_epoch in range(self.config.num_epochs_meta):
            for i in range(1, n):
                prev_data_loader = DataLoader(dataset=data_sequence[i-1], batch_size=self.config.batch_size,
                                              shuffle=True)
                prev_grads = self.main_net.compute_avg_gradient(prev_data_loader)
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

                        imp = self.meta_net.model(meta_inputs)
                        cur_grads += self.config.alpha * imp * (cur_weights - prev_weights)
                        self.main_net.apply_gradient(cur_grads)
                        self.main_net.optimizer.step()
                        self.meta_net.train_single_batch(meta_inputs, joint_grads)
                        self.main_net.compute_loss(cur_inputs, cur_labels)


    def test(self, train_data_sequence, test_data_sequence):
        self.main_net.set_writer(self.config.main_test_writer, self.config.log_dir)
        n = len(train_data_sequence)
        self.acc_matrix = torch.zeros(n, n)

        self.main_net.train(DataLoader(dataset=train_data_sequence[0], batch_size=self.config.batch_size, shuffle=True))
        self.acc_matrix[0][0] = self.main_net.test(DataLoader(dataset=test_data_sequence[0],
                                                              batch_size=self.config.batch_size,
                                                              shuffle=True))
        for i in range(1, n):
            prev_data_loader = DataLoader(dataset=train_data_sequence[i-1], batch_size=self.config.batch_size,
                                          shuffle=True)
            prev_grads = self.main_net.abs_sum_of_gradient(prev_data_loader)
            prev_weights = self.main_net.get_model_weight()
            cur_data_loader = DataLoader(dataset=train_data_sequence[i], batch_size=self.config.batch_size,
                                         shuffle=True)

            for main_epoch in range(self.config.num_epochs_per_task):
                for cur_inputs, cur_labels in cur_data_loader:
                    for v in [cur_inputs, cur_labels]:
                        v = v.to(self.device)
                    cur_grads = self.main_net.compute_gradient(cur_inputs, cur_labels)
                    cur_weights = self.main_net.get_model_weight()
                    meta_inputs = torch.cat((prev_grads, cur_grads, cur_weights), dim=0)

                    imp = self.meta_net.model(meta_inputs)
                    cur_grads += self.config.alpha * imp * (cur_weights - prev_weights)
                    self.main_net.apply_gradient(cur_grads)
                    self.main_net.optimizer.step()
                    self.main_net.compute_loss(cur_inputs, cur_labels)

            for j in range(i+1):
                self.acc_matrix[i][j] = self.main_net.test(DataLoader(dataset=test_data_sequence[j],
                                                                      batch_size=self.config.batch_size,
                                                                      shuffle=True))

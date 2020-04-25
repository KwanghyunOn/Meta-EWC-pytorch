import torch


class MetricTracker:
    def __init__(self, acc_matrix):
        self.acc_matrix = acc_matrix
        self.n_task = self.acc_matrix.shape[0]

    def final_avg_acc(self):
        return self.acc_matrix[-1].mean()

    def total_avg_acc(self):
        return self.acc_matrix.mean()

    def final_forget(self):
        idx = torch.arange(self.n_task)
        return (self.acc_matrix[-1] - self.acc_matrix[idx, idx])[:-1].abs().mean()

    def total_forget(self):
        idx = torch.arange(self.n_task)
        forget = (self.acc_matrix - self.acc_matrix[idx, idx]).abs().tril(diagonal=-1)
        n_forget = self.n_task * (self.n_task - 1) / 2.0
        return forget.sum() / n_forget

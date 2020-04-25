import torch


class MetricTracker:
    def __init__(self, acc_matrix):
        self.acc_matrix = acc_matrix
        self.n_task = self.result.shape[0]
        self.n_total = self.n_task * (self.n_task + 1) / 2.0

    def final_avg_acc(self):
        return self.acc_matrix[-1].mean()

    def total_avg_acc(self):
        return self.acc_matrix.mean()

    def final_forget(self):
        idx = torch.arange(self.n_task)
        return (self.acc_matrix[-1] - self.acc_matrix[idx, idx]).abs().mean()

    def total_forget(self):
        idx = torch.arange(self.n_task)
        return (self.acc_matrix - self.acc_matrix[idx, idx]).abs().mean()


if __main__ == "__name__":
    acc_matrix = torch.rand(10,10)
    print(acc_matrix)
    mt = MetricTracker(acc_matrix)
    print(mt.final_avg_acc())
    print(mt.total_avg_acc())
    print(mt.final_forget())
    print(mt.total_forget())


# class AverageForgetting(Metric):
#     def __init__(self, result):
#         super(AverageForgetting, self).__init__(result)
#
#     def compute(self):
#         self.avg_forget = 0
#         for i in range(self.n_task):
#             last = self.n_task-1
#             first = i
#             self.avg_forget += abs(self.result[first, i] - self.result[last, i])
#
#         self.avg_forget = self.avg_forget / (self.n_task-1)
#
#         return self.avg_forget
#
#
# class TotalForgetting(Metric):
#     def __init__(self, result):
#         super(TotalForgetting, self).__init__(result)
#
#     def compute(self):
#         self.tot_forget = 0
#         for i in range(self.n_task):
#             for j in range(0, self.n_task-i-1):
#                 fore = i
#                 back = i+j+1
#                 self.tot_forget += abs(self.result[fore, i] - self.result[back, i])
#
#         self.n_forget_total = self.n_task * (self.n_task - 1) / 2.0
#         self.tot_forget = self.tot_forget / self.n_forget_total
#
#         return self.tot_forget

import torch
import torch.nn as nn
from datasets import dataset
from models import learner, model, network
import config
from models.utils import DataSequenceProducer
from metrics.metric import MetricTracker


if __name__ == "__main__":
    main_model = model.FCN(28*28, 10, [100])
    loss_main = nn.CrossEntropyLoss()
    opt_main = torch.optim.SGD(main_model.parameters(), lr=0.01, momentum=0.9)
    main_net = network.Network(main_model, loss_main, opt_main, log_dir="logs/exp1/base")

    root = "./datasets/"
    seq_len = 5
    perms = [torch.randperm(28*28) for _ in range(seq_len)]
    train_data_sequence = [dataset.RandPermMnist(root, train=True, perm=perms[i]) for i in range(seq_len)]
    test_data_sequence = [dataset.RandPermMnist(root, train=False, perm=perms[i]) for i in range(seq_len)]

    bl = learner.BaseLearner(main_net, config=config.BaseLearnerConfig)
    bl.test(train_data_sequence, test_data_sequence)

    print(bl.acc_matrix)
    mt = MetricTracker(bl.acc_matrix)
    print(mt.final_avg_acc())
    print(mt.total_avg_acc())
    print(mt.final_forget())
    print(mt.total_forget())

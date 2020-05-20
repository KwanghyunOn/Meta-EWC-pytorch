import torch
import torch.nn as nn
from datasets import dataset
from models import learner, model, network
from models.utils import DataSequenceProducer
from metrics.metric import MetricTracker
import config

from pathlib import Path
import shutil


if __name__ == "__main__":
    cfg = config.EWCLearnerConfig()
    log_dir = Path(cfg.log_dir)
    if log_dir.exists() and log_dir.is_dir():
        shutil.rmtree(cfg.log_dir)
        log_dir.mkdir()
    result_path = Path(cfg.result_path)
    if not result_path.parent.exists():
        result_path.parent.mkdir()

    main_model = model.FCN(28*28, 10, [100])
    loss_main = nn.CrossEntropyLoss()
    opt_main = torch.optim.SGD(main_model.parameters(), lr=0.01, momentum=0.9)
    main_net = network.Network(main_model, loss_main, opt_main)

    seq_len = cfg.seq_len
    perms = [torch.randperm(28*28) for _ in range(seq_len)]
    train_data_sequence = [dataset.RandPermMnist(cfg.data_dir, train=True, perm=perms[i]) for i in range(seq_len)]
    test_data_sequence = [dataset.RandPermMnist(cfg.data_dir, train=False, perm=perms[i]) for i in range(seq_len)]

    el = learner.EWCLearner(main_net, config=cfg)
    el.test(train_data_sequence, test_data_sequence)

    with result_path.open(mode='w') as fw:
        mt = MetricTracker(el.acc_matrix)
        print(el.acc_matrix, file=fw)
        print("final average accuracy: ", mt.final_avg_acc(), file=fw)
        print("total average accuracy: ", mt.total_avg_acc(), file=fw)
        print("final forget: ", mt.final_forget(), file=fw)
        print("total forget: ", mt.total_forget(), file=fw)

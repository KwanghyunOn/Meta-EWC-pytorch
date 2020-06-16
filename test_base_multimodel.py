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
    cfg = config.BaseMultimodelLearnerConfig()
    log_dir = Path(cfg.log_dir)
    if log_dir.exists() and log_dir.is_dir():
        shutil.rmtree(cfg.log_dir)
        log_dir.mkdir()
    result_path = Path(cfg.result_path)
    if not result_path.parent.exists():
        result_path.parent.mkdir()

    n = cfg.seq_len
    main_models = [model.FCN(28*28, 10, [100]) for _ in range(n)]
    loss_main = nn.CrossEntropyLoss()
    opt_mains = [torch.optim.SGD(main_models[i].parameters(), lr=cfg.lr) for i in range(n)]
    main_nets = [network.Network(main_models[i], loss_main, opt_mains[i]) for i in range(n)]

    perms = [torch.randperm(28*28) for _ in range(n)]
    train_data_sequence = [dataset.RandPermMnist(cfg.data_dir, train=True, perm=perms[i]) for i in range(n)]
    test_data_sequence = [dataset.RandPermMnist(cfg.data_dir, train=False, perm=perms[i]) for i in range(n)]

    bl = learner.BaseMultimodelLearner(main_nets, config=cfg)
    bl.test(train_data_sequence, test_data_sequence)

    with result_path.open(mode='w') as fw:
        mt = MetricTracker(bl.acc_matrix)
        print(bl.acc_matrix, file=fw)
        print("final average accuracy: ", mt.final_avg_acc(), file=fw)
        print("total average accuracy: ", mt.total_avg_acc(), file=fw)
        print("final forget: ", mt.final_forget(), file=fw)
        print("total forget: ", mt.total_forget(), file=fw)

import torch
import torch.nn as nn
from datasets import dataset
from models import learner, model, network
import config
from models.utils import DataSequenceProducer
from metrics.metric import MetricTracker

from pathlib import Path
import shutil

if __name__ == "__main__":
    cfg = config.MetaLearnerConfig()
    log_dir = Path(cfg.log_dir)
    if log_dir.exists() and log_dir.is_dir():
        shutil.rmtree(cfg.log_dir)
        log_dir.mkdir()
    result_path = Path(cfg.result_path)
    if not result_path.parent.exists():
        result_path.parent.mkdir()

    main_model = model.FCN(28*28, 10, [100])
    p = 0
    for param in main_model.parameters():
        p += param.data.nelement()

    meta_model = model.FCN(3*p, p, [100])
    p_meta = 0
    for param in meta_model.parameters():
        p_meta += param.data.nelement()

    loss_main = nn.CrossEntropyLoss()
    opt_main = torch.optim.SGD(main_model.parameters(), lr=0.01, momentum=0.9)
    main_net = network.Network(main_model, loss_main, opt_main)

    loss_meta = nn.MSELoss()
    opt_meta = torch.optim.SGD(meta_model.parameters(), lr=0.01, momentum=0.9)
    meta_net = network.Network(meta_model, loss_meta, opt_meta)

    seq_len = cfg.seq_len
    meta_perms = [torch.randperm(28*28) for _ in range(seq_len)]
    perms = [torch.randperm(28*28) for _ in range(seq_len)]
    # meta_train_data_sequence = [dataset.RandPermMnist(cfg.data_dir, train=True, perm=meta_perms[i]) for i in range(seq_len)]
    meta_train_data_sequence = [dataset.RandPermMnist(cfg.data_dir, train=True, perm=perms[i]) for i in range(seq_len)]
    train_data_sequence = [dataset.RandPermMnist(cfg.data_dir, train=True, perm=perms[i]) for i in range(seq_len)]
    test_data_sequence = [dataset.RandPermMnist(cfg.data_dir, train=False, perm=perms[i]) for i in range(seq_len)]

    ml = learner.MetaLearner(main_net, meta_net, config=cfg)
    ml.train(meta_train_data_sequence)
    ml.test(train_data_sequence, test_data_sequence)

    with result_path.open(mode='w') as fw:
        mt = MetricTracker(ml.acc_matrix)
        print(ml.acc_matrix, file=fw)
        print("final average accuracy: ", mt.final_avg_acc(), file=fw)
        print("total average accuracy: ", mt.total_avg_acc(), file=fw)
        print("final forget: ", mt.final_forget(), file=fw)
        print("total forget: ", mt.total_forget(), file=fw)

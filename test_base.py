import torch
import torch.nn as nn
from datasets import dataset
from models import learner, model, network
from models.utils import DataSequenceProducer
from metrics.metric import MetricTracker
import config

from pathlib import Path
import shutil
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.1)
    parser.add_argument('--n', default=1)
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    
    cfg = config.BaseLearnerConfig(float(args.lr), args.name)
    log_dir = Path(cfg.log_dir)
    if log_dir.exists() and log_dir.is_dir():
        shutil.rmtree(cfg.log_dir)
        log_dir.mkdir()
    result_path = Path(cfg.result_path)
    if not result_path.parent.exists():
        result_path.parent.mkdir()

    am, faa, taa, ff, tf = [], [], [], [], []
    for _ in range(int(args.n)):
        main_model = model.FCN(28*28, 10, [50, 50])
        loss_main = nn.CrossEntropyLoss()
        opt_main = torch.optim.SGD(main_model.parameters(), lr=cfg.lr)
        main_net = network.Network(main_model, loss_main, opt_main)

        seq_len = cfg.seq_len
        perms = [torch.randperm(28*28) for _ in range(seq_len)]
        train_data_sequence = [dataset.RandPermMnist(cfg.data_dir, train=True, perm=perms[i]) for i in range(seq_len)]
        test_data_sequence = [dataset.RandPermMnist(cfg.data_dir, train=False, perm=perms[i]) for i in range(seq_len)]

        bl = learner.BaseLearner(main_net, config=cfg)
        bl.test(train_data_sequence, test_data_sequence) 

        mt = MetricTracker(bl.acc_matrix)
        am.append(bl.acc_matrix)
        faa.append(mt.final_avg_acc())
        taa.append(mt.total_avg_acc())
        ff.append(mt.final_forget())
        tf.append(mt.total_forget())

    am = torch.stack(am)
    faa = torch.stack(faa)
    taa = torch.stack(taa)
    ff = torch.stack(ff)
    tf = torch.stack(tf)

    with result_path.open(mode='w') as fw:
        print(f"lr = {cfg.lr:.3f}, batch size = {cfg.batch_size}", file=fw)
        print(f"seq len = {cfg.seq_len}, exp num = {int(args.n)}", file=fw)
        print(torch.mean(am, dim=0), file=fw)
        print(f"final average acc: {torch.mean(faa):.3f}, std: {torch.std(faa):.3f}", file=fw)
        print(f"total average acc: {torch.mean(taa):.3f}, std: {torch.std(taa):.3f}", file=fw)
        print(f"final forget: {torch.mean(ff):.3f}, std: {torch.std(ff):.3f}", file=fw)
        print(f"total forget: {torch.mean(tf):.3f}, std: {torch.std(tf):.3f}", file=fw)

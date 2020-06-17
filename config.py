
class Config:
    data_dir = "datasets/"

class BaseLearnerConfig(Config):
    def __init__(self, lr, name):
        self.lr = lr
        self.seq_len = 10
        self.batch_size = 256
        self.num_epochs_per_task = 1
        self.writer = "Loss/train-main"
        self.log_dir = f"logs/{name}/base_lr{lr}/"
        self.result_path = f"results/{name}/base_lr{lr}.txt"

class BaseJointLearnerConfig(Config):
    lr = 0.4
    seq_len = 10
    batch_size = 256
    num_epochs_per_task = 1
    writer = "Loss/train-main"
    log_dir = "logs/exp2/base_joint/"
    result_path = "results/exp2/base_joint.txt"

class BaseMultimodelLearnerConfig(Config):
    lr = 0.4
    seq_len = 10
    batch_size = 256
    num_epochs_per_task = 1
    writer = "Loss/train-main"
    log_dir = "logs/exp2/base_multimodel/"
    result_path = "results/exp2/base_multimodel.txt"

class EWCLearnerConfig(Config):
    def __init__(self, lr, alpha, name):
        self.lr = lr
        self.alpha = alpha
        self.seq_len = 10
        self.batch_size = 256
        self.num_epochs_per_task = 1
        self.writer = "Loss/train-main"
        self.log_dir = f"logs/{name}/EWC_lr{lr}_alpha{alpha}"
        self.result_path = f"results/{name}/EWC_lr{lr}_alpha{alpha}.txt"

class MetaLearnerConfig(Config):
    def __init__(self, lr_meta, lr_main, alpha, eps, name):
        self.lr_meta = lr_meta
        self.lr_main = lr_main
        self.alpha = alpha
        self.eps = eps
        self.seq_len = 3
        self.batch_size = 256
        self.num_epochs_meta = 1
        self.num_epochs_main = 1
        self.num_epochs_per_task = 1
        self.num_warmup = 1
        self.meta_train_writer = "Loss/train-meta"
        self.main_train_writer = "Loss/train-main"
        self.main_test_writer = "Loss/test-main"
        self.log_dir = f"logs/{name}/meta_lrmeta{lr_meta}_lrmain{lr_main}_alpha{alpha}"
        self.result_path = f"results/{name}/meta_lrmeta{lr_meta}_lrmain{lr_main}_alpha{alpha}.txt"


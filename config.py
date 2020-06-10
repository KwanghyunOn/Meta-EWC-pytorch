
class Config:
    data_dir = "datasets/"

class BaseLearnerConfig(Config):
    seq_len = 10
    batch_size = 256
    num_epochs_per_task = 1
    writer = "Loss/train-main"
    log_dir = "logs/exp1/base/"
    result_path = "results/exp1/base.txt"

class EWCLearnerConfig(Config):
    seq_len = 10
    batch_size = 256
    alpha = 0.1
    num_epochs_per_task = 1
    writer = "Loss/train-main"
    log_dir = "logs/exp1/EWC/"
    result_path = "results/exp1/EWC.txt"

class MetaLearnerConfig(Config):
    seq_len = 10
    batch_size = 256
    alpha = 0.1
    num_epochs_meta = 1
    num_epochs_main = 1
    num_epochs_per_task = 1
    num_warmup = 1
    meta_train_writer = "Loss/train-meta"
    main_train_writer = "Loss/train-main"
    main_test_writer = "Loss/test-main"
    log_dir = "logs/exp1/meta_same"
    result_path = "results/exp1/meta_same.txt"


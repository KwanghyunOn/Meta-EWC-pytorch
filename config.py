
class Config:
    data_dir = "datasets/"
    log_dir = "logs/exp1/EWC/"
    result_path = "results/exp1/EWC.txt"

class BaseLearnerConfig(Config):
    seq_len = 10
    batch_size = 256
    num_epochs_per_task = 1

class EWCLearnerConfig(Config):
    seq_len = 10
    batch_size = 256
    alpha = 0.1
    num_epochs_per_task = 1

class MetaLearnerConfig(Config):
    seq_len = 10
    batch_size = 256
    alpha = 0.1
    num_epochs_meta = 1
    num_epochs_main = 1
    num_epochs_per_task = 1


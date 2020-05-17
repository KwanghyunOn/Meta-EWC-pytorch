
class Config:
    batch_size = 10

class BaseLearnerConfig(Config):
    batch_size = 256
    alpha = 0.1
    num_epochs_main = 1
    num_epochs_per_task = 1

class MetaLearnerConfig(Config):
    batch_size = 256
    alpha = 0.1
    num_epochs_meta = 1
    num_epochs_main = 1
    num_epochs_per_task = 1


from easydict import EasyDict

from configs.model_config import cfg as model_cfg

cfg = EasyDict()

cfg.model_cfg = model_cfg

cfg.batch_size = 128
cfg.lr = 1e-4
cfg.weight_decay = 1e-4
cfg.epochs = 100
cfg.reg_lambda = 1e-4

cfg.log_metrics = False
cfg.experiment_name = 'undercomplete_ae'

cfg.evaluate_on_train_set = False
cfg.evaluate_before_training = True
cfg.eval_plots_dir = f'../saved_files/plots/{cfg.experiment_name}/'

cfg.load_saved_model = True
cfg.checkpoints_dir = f'../saved_files/checkpoints/{cfg.experiment_name}'
cfg.epoch_to_load = 99
cfg.save_model = False
cfg.epochs_saving_freq = 1

cfg.save_vectors = False
cfg.vectors_path = f'../saved_files/vectors/{cfg.experiment_name}/'

cfg.run_knn = True
cfg.run_lof = True

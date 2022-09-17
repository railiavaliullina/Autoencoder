from easydict import EasyDict
import numpy as np

from configs.train_config import cfg as train_cfg

cfg = EasyDict()
cfg.k_list = np.arange(1, 101)

cfg.load_saved_dists = True
cfg.calculate_dists_in_loop = True

cfg.saved_dists_path = f'../saved_files/dists/{train_cfg.experiment_name}/cifar_dists.pickle'
cfg.saved_vectors_path = f'../saved_files/vectors/{train_cfg.experiment_name}/cifar_vectors.pickle'
cfg.plots_dir = f'../saved_files/plots/{train_cfg.experiment_name}/'

from easydict import EasyDict

cfg = EasyDict()

cfg.data_path = '../data'
cfg.anomaly_class = 'airplane'

cfg.mean = [0.485, 0.456, 0.406]
cfg.std = [0.229, 0.224, 0.225]

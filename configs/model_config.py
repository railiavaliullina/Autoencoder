from easydict import EasyDict

from enums.AutoencoderType import AEType
from enums.RegType import RegType


cfg = EasyDict()

cfg.overcomplete = EasyDict()
cfg.undercomplete = EasyDict()

cfg.undercomplete.k_size = 4
cfg.undercomplete.padding = 1
cfg.undercomplete.stride = 2

cfg.overcomplete.k_size = 3
cfg.overcomplete.padding = 1
cfg.overcomplete.stride = 1

cfg.autoencoder_type = AEType.undercomplete
cfg.reg_type = RegType.sparse

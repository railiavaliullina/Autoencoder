from torch import nn

from enums.AutoencoderType import AEType


class AE(nn.Module):
    def __init__(self, cfg):
        super(AE, self).__init__()

        self.cfg = cfg

        if self.cfg.autoencoder_type == AEType.undercomplete:
            self.get_undercomplete_ae_architecture()

        elif self.cfg.autoencoder_type == AEType.overcomplete:
            self.get_overcomplete_ae_architecture()
        else:
            raise Exception

        self.get_parameters_num()

    def get_undercomplete_ae_architecture(self):
        cfg = self.cfg.undercomplete

        # encoder
        self.l_relu = nn.LeakyReLU()
        self.conv_1 = nn.Conv2d(3, 64, kernel_size=cfg.k_size, padding=cfg.padding, stride=cfg.stride)
        self.conv_3 = nn.Conv2d(64, 128, kernel_size=cfg.k_size, padding=cfg.padding, stride=cfg.stride)
        self.BN_4 = nn.BatchNorm2d(128)
        self.conv_6 = nn.Conv2d(128, 256, kernel_size=cfg.k_size, padding=cfg.padding, stride=cfg.stride)
        self.BN_7 = nn.BatchNorm2d(256)
        self.conv_9 = nn.Conv2d(256, 100, kernel_size=cfg.k_size)
        self.encoder = nn.Sequential(self.conv_1, self.l_relu, self.conv_3, self.BN_4, self.l_relu, self.conv_6,
                                     self.BN_7, self.l_relu, self.conv_9)
        # decoder
        self.relu = nn.ReLU()
        self.transposed_conv_1 = nn.ConvTranspose2d(100, 256, kernel_size=cfg.k_size)
        self.BN_2 = nn.BatchNorm2d(256)
        self.transposed_conv_4 = nn.ConvTranspose2d(256, 128, kernel_size=cfg.k_size, padding=cfg.padding,
                                                    stride=cfg.stride)
        self.BN_5 = nn.BatchNorm2d(128)
        self.transposed_conv_7 = nn.ConvTranspose2d(128, 64, kernel_size=cfg.k_size, padding=cfg.padding,
                                                    stride=cfg.stride)
        self.BN_8 = nn.BatchNorm2d(64)
        self.transposed_conv_10 = nn.ConvTranspose2d(64, 3, kernel_size=cfg.k_size, padding=cfg.padding,
                                                     stride=cfg.stride)
        self.sigmoid_11 = nn.Sigmoid()
        self.decoder = nn.Sequential(self.transposed_conv_1, self.BN_2, self.relu, self.transposed_conv_4, self.BN_5,
                                     self.relu, self.transposed_conv_7, self.BN_8, self.relu, self.transposed_conv_10,
                                     self.sigmoid_11)

    def get_overcomplete_ae_architecture(self):
        cfg = self.cfg.overcomplete

        # encoder
        self.l_relu = nn.LeakyReLU()
        self.conv_0 = nn.Conv2d(3, 64, kernel_size=cfg.k_size, padding=cfg.padding, stride=cfg.stride)
        self.conv_1 = nn.Conv2d(64, 64, kernel_size=cfg.k_size, padding=cfg.padding, stride=cfg.stride)
        self.BN_1 = nn.BatchNorm2d(64)
        self.conv_2 = nn.Conv2d(64, 128, kernel_size=cfg.k_size, padding=cfg.padding, stride=cfg.stride)
        self.BN_2 = nn.BatchNorm2d(128)
        self.conv_3 = nn.Conv2d(128, 256, kernel_size=cfg.k_size, padding=cfg.padding, stride=cfg.stride)
        self.BN_3 = nn.BatchNorm2d(256)
        self.conv_4 = nn.Conv2d(256, 256, kernel_size=cfg.k_size, padding=cfg.padding, stride=cfg.stride)
        self.BN_4 = nn.BatchNorm2d(256)
        self.conv_5 = nn.Conv2d(256, 3, kernel_size=cfg.k_size, padding=cfg.padding, stride=cfg.stride)

        self.encoder = nn.Sequential(self.conv_0, self.l_relu,
                                     self.conv_1, self.BN_1, self.l_relu,
                                     self.conv_2, self.BN_2, self.l_relu,
                                     self.conv_3, self.BN_3, self.l_relu,
                                     self.conv_4, self.BN_4, self.l_relu,
                                     self.conv_5)

        # decoder
        self.relu = nn.ReLU()
        self.transposed_conv_0 = nn.ConvTranspose2d(3, 256, kernel_size=cfg.k_size, padding=cfg.padding,
                                                    stride=cfg.stride)
        self.BN_0_d = nn.BatchNorm2d(256)
        self.transposed_conv_1 = nn.ConvTranspose2d(256, 256, kernel_size=cfg.k_size, padding=cfg.padding,
                                                    stride=cfg.stride)
        self.BN_1_d = nn.BatchNorm2d(256)
        self.transposed_conv_2 = nn.ConvTranspose2d(256, 128, kernel_size=cfg.k_size, padding=cfg.padding,
                                                    stride=cfg.stride)
        self.BN_2_d = nn.BatchNorm2d(128)
        self.transposed_conv_3 = nn.ConvTranspose2d(128, 64, kernel_size=cfg.k_size, padding=cfg.padding,
                                                    stride=cfg.stride)
        self.BN_3_d = nn.BatchNorm2d(64)
        self.transposed_conv_4 = nn.ConvTranspose2d(64, 64, kernel_size=cfg.k_size, padding=cfg.padding,
                                                    stride=cfg.stride)
        self.BN_4_d = nn.BatchNorm2d(64)
        self.transposed_conv_5 = nn.ConvTranspose2d(64, 3, kernel_size=cfg.k_size, padding=cfg.padding,
                                                    stride=cfg.stride)
        self.sigmoid = nn.Sigmoid()

        self.decoder = nn.Sequential(self.transposed_conv_0, self.BN_0_d,
                                     self.transposed_conv_1, self.BN_1_d, self.relu,
                                     self.transposed_conv_2, self.BN_2_d, self.relu,
                                     self.transposed_conv_3, self.BN_3_d, self.relu,
                                     self.transposed_conv_4, self.BN_4_d, self.relu,
                                     self.transposed_conv_5,
                                     self.sigmoid)

    def get_parameters_num(self):
        encoder_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        decoder_params = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        all_params = encoder_params + decoder_params
        print(f'\nAutoencoder type: {self.cfg.autoencoder_type.name}'
              f'\nEncoder trainable params num: {encoder_params}'
              f'\nDecoder trainable params num: {decoder_params}'
              f'\nModel trainable params num: {all_params}\n')

    def forward(self, x):
        h = self.encoder(x)
        x = self.decoder(h)
        return x, h

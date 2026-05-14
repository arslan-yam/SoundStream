from torch import nn

from src.model.discriminator import Discriminator
from src.model.soundstream import SoundStream


class SoundStreamModel(nn.Module):
    def __init__(self, C=32, D=128, strides=(2, 4, 5, 5), n_q=8, codebook_size=1024,
        decay=0.99, n_scales=3, n_fft=1024, hop_length=256):
        super().__init__()
        self.generator = SoundStream(C=C, D=D, strides=strides, n_q=n_q, codebook_size=codebook_size, decay=decay)
        self.discriminator = Discriminator(n_scales=n_scales, n_fft=n_fft, hop_length=hop_length)

    def forward(self, audio, **batch):
        return self.generator(audio)

    def generator_parameters(self):
        return self.generator.generator_parameters()

    def discriminator_parameters(self):
        return list(self.discriminator.parameters())

    def __str__(self):
        all_parameters = sum([p.numel() for p in self.parameters()])
        g_params = sum([p.numel() for p in self.generator.parameters()])
        d_params = sum([p.numel() for p in self.discriminator.parameters()])
        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nGenerator parameters: {g_params}"
        result_info = result_info + f"\nDiscriminator parameters: {d_params}"
        return result_info

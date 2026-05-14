import torch
from torch import nn
from torch.nn.utils import weight_norm


class WaveScaleDiscriminator(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Sequential(
                weight_norm(nn.Conv1d(1, 16, kernel_size=15, padding=7)),
                nn.LeakyReLU(0.2),
            ),
            nn.Sequential(
                weight_norm(nn.Conv1d(16, 64, kernel_size=41, stride=4, groups=4, padding=20)),
                nn.LeakyReLU(0.2),
            ),
            nn.Sequential(
                weight_norm(nn.Conv1d(64, 256, kernel_size=41, stride=4, groups=16, padding=20)),
                nn.LeakyReLU(0.2),
            ),
            nn.Sequential(
                weight_norm(nn.Conv1d(256, 1024, kernel_size=41, stride=4, groups=64, padding=20)),
                nn.LeakyReLU(0.2),
            ),
            nn.Sequential(
                weight_norm(nn.Conv1d(1024, 1024, kernel_size=41, stride=4, groups=256, padding=20)),
                nn.LeakyReLU(0.2),
            ),
            nn.Sequential(
                weight_norm(nn.Conv1d(1024, 1024, kernel_size=5, padding=2)),
                nn.LeakyReLU(0.2),
            ),
            weight_norm(nn.Conv1d(1024, 1, kernel_size=3, padding=1)),
        )

    def forward(self, x):
        feats = []
        for block in self.net:
            x = block(x)
            feats.append(x)
        return x, feats


class MultiScaleWaveDiscriminator(nn.Module):
    def __init__(self, n_scales=3):
        super().__init__()
        self.discriminators = nn.ModuleList([WaveScaleDiscriminator() for _ in range(n_scales)])
        self.pool = nn.AvgPool1d(kernel_size=4, stride=2, padding=2)

    def forward(self, x):
        outs, feats_list = [], []
        for i, d in enumerate(self.discriminators):
            if i > 0:
                x = self.pool(x)
            out, feats = d(x)
            outs.append(out)
            feats_list.append(feats)
        return outs, feats_list


class STFTDiscriminator(nn.Module):

    def __init__(self, n_fft=1024, hop_length=256):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft))
        ch = 32

        self.net = nn.Sequential(
            nn.Sequential(
                weight_norm(nn.Conv2d(2, ch, kernel_size=(7, 7), padding=(3, 3))),
                nn.LeakyReLU(0.2),
            ),
            nn.Sequential(
                weight_norm(nn.Conv2d(ch, ch, kernel_size=(3, 8), stride=(1, 2), padding=(1, 3))),
                nn.LeakyReLU(0.2),
            ),
            nn.Sequential(
                weight_norm(nn.Conv2d(ch, ch, kernel_size=(3, 8), stride=(1, 2), padding=(1, 3))),
                nn.LeakyReLU(0.2),
            ),
            nn.Sequential(
                weight_norm(nn.Conv2d(ch, 2 * ch, kernel_size=(3, 8), stride=(1, 2), padding=(1, 3))),
                nn.LeakyReLU(0.2),
            ),
            nn.Sequential(
                weight_norm(nn.Conv2d(2 * ch, 2 * ch, kernel_size=(3, 8), stride=(1, 2), padding=(1, 3))),
                nn.LeakyReLU(0.2),
            ),
            nn.Sequential(
                weight_norm(nn.Conv2d(2 * ch, 4 * ch, kernel_size=(3, 8), stride=(1, 2), padding=(1, 3))),
                nn.LeakyReLU(0.2),
            ),
            weight_norm(nn.Conv2d(4 * ch, 1, kernel_size=(3, 3), padding=(1, 1))),
        )

    def forward(self, x):
        x = x.squeeze(1)
        stft = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, return_complex=True, center=True)
        h = torch.stack([stft.real, stft.imag], dim=1)
        feats = []
        for block in self.net:
            h = block(h)
            feats.append(h)
        return h, feats


class Discriminator(nn.Module):
    def __init__(self, n_scales=3, n_fft=1024, hop_length=256):
        super().__init__()
        self.msd = MultiScaleWaveDiscriminator(n_scales=n_scales)
        self.stftd = STFTDiscriminator(n_fft=n_fft, hop_length=hop_length)

    def forward(self, x):
        msd_outs, msd_feats = self.msd(x)
        stft_out, stft_feats = self.stftd(x)
        return msd_outs + [stft_out], msd_feats + [stft_feats]

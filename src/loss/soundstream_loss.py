import torch
import torch.nn.functional as F
import torchaudio
from torch import nn


class MultiScaleMelLoss(nn.Module):
    def __init__(self, sample_rate=16000, scales=(5, 6, 7, 8, 9, 10, 11), n_mels=64):
        super().__init__()
        self.scales = scales
        self.mels = nn.ModuleList()
        for s in scales:
            mel = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=2**s, win_length=2**s, hop_length=max(2**s // 4, 1), n_mels=n_mels, power=1.0)
            self.mels.append(mel)

    def forward(self, x, x_hat):
        x = x.squeeze(1)
        x_hat = x_hat.squeeze(1)
        loss = 0.0
        for s, mel in zip(self.scales, self.mels):
            m_x, m_hat = mel(x), mel(x_hat)
            l1 = F.l1_loss(m_x, m_hat)
            log_l2 = F.mse_loss(torch.log(m_x + 1e-5), torch.log(m_hat + 1e-5)).sqrt()
            loss = loss + l1 + (2**s / 2) ** 0.5 * log_l2
        return loss


class SoundStreamLoss(nn.Module):
    def __init__(self, sample_rate=16000, lambda_adv=1.0, lambda_feat=100.0, lambda_rec=1.0, lambda_commit=1.0):
        super().__init__()
        self.rec_loss = MultiScaleMelLoss(sample_rate=sample_rate)
        self.lambda_adv = lambda_adv
        self.lambda_feat = lambda_feat
        self.lambda_rec = lambda_rec
        self.lambda_commit = lambda_commit

    def discriminator_loss(self, real_outs, fake_outs):
        loss = 0.0
        for r, f in zip(real_outs, fake_outs):
            loss = loss + F.relu(1.0 - r).mean() + F.relu(1.0 + f).mean()
        return loss

    def adversarial_loss(self, fake_outs):
        loss = 0.0
        for f in fake_outs:
            loss = loss + F.relu(1.0 - f).mean()
        return loss

    def feature_matching_loss(self, real_feats, fake_feats):
        loss = 0.0
        cnt = 0
        for rf_group, ff_group in zip(real_feats, fake_feats):
            for r, f in zip(rf_group, ff_group):
                loss = loss + F.l1_loss(f, r.detach())
                cnt += 1
        return loss / max(cnt, 1)

    def forward(self, **kwargs):
        return {}

import torch
import torch.nn.functional as F
from torch import nn


class Crop1d(nn.Module):
    def __init__(self, left, right):
        super().__init__()
        self.left = left
        self.right = right

    def forward(self, x):
        return x[..., self.left : x.shape[-1] - self.right]

def get_padding(kernel_size, dilation=1):
    return ((kernel_size - 1) * dilation) // 2

class ResidualUnit(nn.Module):

    def __init__(self, channels, dilation):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=7, dilation=dilation, padding=get_padding(7, dilation)),
            nn.ELU(),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.ELU(),
        )

    def forward(self, x):
        return x + self.net(x)


class EncoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        pad_left = stride // 2
        pad_right = stride - pad_left
        self.net = nn.Sequential(
            ResidualUnit(in_channels, dilation=1),
            ResidualUnit(in_channels, dilation=3),
            ResidualUnit(in_channels, dilation=9),
            nn.ELU(),
            nn.ConstantPad1d((pad_left, pad_right), 0.0),
            nn.Conv1d(in_channels, out_channels, kernel_size=2 * stride, stride=stride),
        )

    def forward(self, x):
        return self.net(x)


class DecoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        pad_left = stride // 2
        pad_right = stride - pad_left
        self.net = nn.Sequential(
            nn.ELU(),
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2 * stride, stride=stride),
            Crop1d(pad_left, pad_right),
            ResidualUnit(out_channels, dilation=1),
            ResidualUnit(out_channels, dilation=3),
            ResidualUnit(out_channels, dilation=9),
        )

    def forward(self, x):
        return self.net(x)


class Encoder(nn.Module):

    def __init__(self, C=32, D=128, strides=(2, 4, 5, 5)):
        super().__init__()
        layers = [nn.Conv1d(1, C, kernel_size=7, padding=3)]
        in_ch = C
        for i, s in enumerate(strides):
            out_ch = C * (2 ** (i + 1))
            layers.append(EncoderBlock(in_ch, out_ch, s))
            in_ch = out_ch
        layers.append(nn.Conv1d(in_ch, D, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):

    def __init__(self, C=32, D=128, strides=(2, 4, 5, 5)):
        super().__init__()
        n = len(strides)
        in_ch = C * (2**n)
        layers = [nn.Conv1d(D, in_ch, kernel_size=7, padding=3)]
        for s in reversed(strides):
            out_ch = in_ch // 2
            layers.append(DecoderBlock(in_ch, out_ch, s))
            in_ch = out_ch
        layers.append(nn.ELU())
        layers.append(nn.Conv1d(in_ch, 1, kernel_size=7, padding=3))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size, dim, decay=0.99, eps=1e-5):
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = dim
        self.decay = decay
        self.eps = eps
        embed = torch.randn(codebook_size, dim) * 0.01
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embed_avg", embed.clone())
        self.register_buffer("initialized", torch.tensor(False))

    def _init_codebook(self, flat):
        n = flat.shape[0]
        idx = torch.randperm(n, device=flat.device)[: self.codebook_size]
        if idx.shape[0] < self.codebook_size:
            extra = torch.randint(0, n, (self.codebook_size - idx.shape[0],), device=flat.device)
            idx = torch.cat([idx, extra])
        self.embed.data.copy_(flat[idx])
        self.embed_avg.data.copy_(self.embed.data)
        self.cluster_size.data.fill_(1.0)
        self.initialized.fill_(True)

    def forward(self, x):
        x_perm = x.permute(0, 2, 1).contiguous()
        flat = x_perm.view(-1, self.dim)
        if self.training and not bool(self.initialized):
            self._init_codebook(flat)
        dist = (flat.pow(2).sum(1, keepdim=True) - 2 * flat @ self.embed.t() + self.embed.pow(2).sum(1, keepdim=True).t())
        indices = dist.argmin(dim=1)
        onehot = F.one_hot(indices, self.codebook_size).type(flat.dtype)
        quant = onehot @ self.embed

        if self.training:
            embed_sum_new = onehot.t() @ flat
            self.cluster_size.data.mul_(self.decay).add_(onehot.sum(0), alpha=1 - self.decay)
            self.embed_avg.data.mul_(self.decay).add_(embed_sum_new, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            smoothed = ((self.cluster_size + self.eps) / (n + self.codebook_size * self.eps) * n)
            self.embed.data.copy_(self.embed_avg / smoothed.unsqueeze(1))

        quant = quant.view(x_perm.shape).permute(0, 2, 1).contiguous()
        indices = indices.view(x.shape[0], x.shape[2])
        quant_st = x + (quant - x).detach()
        return quant_st, indices, quant


class ResidualVQ(nn.Module):
    def __init__(self, n_q=8, codebook_size=1024, dim=128, decay=0.99):
        super().__init__()
        self.n_q = n_q
        self.codebook_size = codebook_size
        self.quantizers = nn.ModuleList([VectorQuantizer(codebook_size, dim, decay=decay) for i in range(n_q)])

    def forward(self, x):
        resid = x
        qsum = torch.zeros_like(x)
        all_indices = []
        for q in self.quantizers:
            q_st, indices, q_hard = q(resid)
            qsum = qsum + q_st
            resid = resid - q_hard.detach()
            all_indices.append(indices)
        commit_loss = F.mse_loss(x, qsum.detach())
        indices = torch.stack(all_indices, dim=1)
        return qsum, indices, commit_loss

    def decode(self, indices):
        q_sum = 0
        for i, quantizer in enumerate(self.quantizers):
            embed = quantizer.embed[indices[:, i]]
            q_sum = q_sum + embed.permute(0, 2, 1)
        return q_sum


class SoundStream(nn.Module):
    def __init__(self, C=32, D=128, strides=(2, 4, 5, 5), n_q=8, codebook_size=1024, decay=0.99):
        super().__init__()
        self.encoder = Encoder(C=C, D=D, strides=strides)
        self.quantizer = ResidualVQ(n_q=n_q, codebook_size=codebook_size, dim=D, decay=decay)
        self.decoder = Decoder(C=C, D=D, strides=strides)
        self.total_stride = 1
        for s in strides:
            self.total_stride *= s

    def pad_audio(self, audio):
        t = audio.shape[-1]
        rem = t % self.total_stride
        if rem != 0:
            pad = self.total_stride - rem
            audio = F.pad(audio, (0, pad))
        return audio, t

    def forward(self, audio, **batch):
        audio_padded, orig_len = self.pad_audio(audio)
        z = self.encoder(audio_padded)
        z_q, indices, commit_loss = self.quantizer(z)
        audio_pred = self.decoder(z_q)
        audio_pred = audio_pred[..., :orig_len]
        return {
            "audio_pred": audio_pred,
            "z": z,
            "z_q": z_q,
            "indices": indices,
            "commit_loss": commit_loss,
        }

    def generator_parameters(self):
        return (list(self.encoder.parameters()) + list(self.quantizer.parameters()) + list(self.decoder.parameters()))

    def __str__(self):
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )
        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"
        return result_info

import argparse
import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio
from tqdm import tqdm

from src.model import SoundStreamModel

n_fft, hop, n_mels, n_mfcc, max_sec = 1024, 256, 80, 13, 30.0
SR = 16000
win = torch.hann_window(n_fft)
freqs = torch.linspace(0, SR / 2, n_fft // 2 + 1)
mfcc = torchaudio.transforms.MFCC(sample_rate=SR, n_mfcc=n_mfcc, melkwargs={"n_fft": n_fft, "hop_length": hop, "n_mels": n_mels, "power": 2.0})


def load_audio(path):
    data, sr = sf.read(str(path), dtype="float32")
    wav = torch.from_numpy(data)
    if wav.ndim > 1:
        wav = wav.mean(-1)
    if sr != SR:
        wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, SR).squeeze(0)
    return wav[: int(max_sec * SR)]


def stft_mag(x):
    return torch.stft(x, n_fft, hop, window=win, return_complex=True).abs()


def lsd(real, pred, eps=1e-8):
    sr_db = 20 * torch.log10(stft_mag(real) + eps)
    sp_db = 20 * torch.log10(stft_mag(pred) + eps)
    return torch.sqrt(((sr_db - sp_db) ** 2).mean(0)).mean().item()


def mcd(real, pred):
    diff = (mfcc(real)[1:] - mfcc(pred)[1:]) ** 2
    return ((10 / math.log(10)) * torch.sqrt(2 * diff.sum(0))).mean().item()


def band_energy_db(x):
    P = stft_mag(x) ** 2
    masks = [freqs < 500, (freqs >= 500) & (freqs < 4000), freqs >= 4000]
    return tuple(10 * math.log10(P[m].mean().item() + 1e-12) for m in masks)


def spectral_centroid(x, eps=1e-10):
    S = stft_mag(x)
    c = (freqs.unsqueeze(-1) * S).sum(0) / (S.sum(0) + eps)
    return c.mean().item(), c.std().item()


def spectral_rolloff(x, pct):
    cum = (stft_mag(x) ** 2).cumsum(0)
    idx = (cum >= pct * cum[-1:]).int().argmax(0)
    return freqs[idx].mean().item()


def spectral_flatness(x, eps=1e-10):
    S = stft_mag(x) + eps
    return (S.log().mean(0).exp() / (S.mean(0) + eps)).mean().item()


def kl(p, q, eps=1e-10):
    return float(np.sum(p * np.log((p + eps) / (q + eps))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--n_files", type=int, default=200)
    ap.add_argument("--checkpoint", default="saved/ss_5090_run1/model_best.pth")
    ap.add_argument("--comet", action="store_true")
    ap.add_argument("--workspace", default="arslan-yamaletdinov")
    ap.add_argument("--project", default="dl-big-hw")
    ap.add_argument("--run_name", default="ss_stats")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SoundStreamModel().to(device).eval()
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
    files = sorted(str(p) for p in Path(args.data_dir).rglob("*") if p.suffix.lower() in {".flac", ".wav", ".mp3", ".ogg"})
    np.random.seed(42)
    np.random.shuffle(files)
    files = files[: args.n_files]
    bins = np.linspace(-1, 1, 101)
    real_hist = np.zeros(100)
    pred_hist = np.zeros(100)
    rows = []

    for path in tqdm(files):
        try:
            real = load_audio(path)
        except Exception:
            continue
        with torch.no_grad():
            pred = model(audio=real[None, None].to(device))["audio_pred"]
        pred = pred.squeeze().cpu().clamp(-1, 1)
        real = real[: pred.numel()]
        be_r = band_energy_db(real)
        be_p = band_energy_db(pred)
        c_r_mean, c_r_std = spectral_centroid(real)
        c_p_mean, c_p_std = spectral_centroid(pred)

        rows.append({
            "SI_SDR_dB": scale_invariant_signal_distortion_ratio(pred, real).item(),
            "LSD_dB": lsd(real, pred),
            "MCD_dB": mcd(real, pred),
            "low_real_dB": be_r[0], "low_pred_dB": be_p[0],
            "mid_real_dB": be_r[1], "mid_pred_dB": be_p[1],
            "high_real_dB": be_r[2], "high_pred_dB": be_p[2],
            "centroid_real_Hz": c_r_mean, "centroid_pred_Hz": c_p_mean,
            "centroid_std_real": c_r_std, "centroid_std_pred": c_p_std,
            "rolloff85_real_Hz": spectral_rolloff(real, 0.85),
            "rolloff85_pred_Hz": spectral_rolloff(pred, 0.85),
            "rolloff95_real_Hz": spectral_rolloff(real, 0.95),
            "rolloff95_pred_Hz": spectral_rolloff(pred, 0.95),
            "flatness_real": spectral_flatness(real),
            "flatness_pred": spectral_flatness(pred),
        })
        real_hist += np.histogram(real.numpy(), bins=bins)[0]
        pred_hist += np.histogram(pred.numpy(), bins=bins)[0]

    if not rows:
        return

    p = real_hist / (real_hist.sum() + 1e-12)
    q = pred_hist / (pred_hist.sum() + 1e-12)
    kl_val = kl(p, q)
    js_val = 0.5 * kl(p, 0.5 * (p + q)) + 0.5 * kl(q, 0.5 * (p + q))
    summary = {}
    for k in rows[0]:
        v = np.array([r[k] for r in rows])
        summary[k] = (v.mean(), v.std(), float(np.median(v)))

    if args.comet:
        import comet_ml
        exp = comet_ml.Experiment(
            workspace=args.workspace, project_name=args.project,
            log_code=False, log_graph=False,
            auto_metric_logging=False, auto_param_logging=False,
        )
        exp.set_name(args.run_name)
        exp.log_parameters({"data_dir": args.data_dir, "n_files": len(rows)})
        for k, (m, s, med) in summary.items():
            exp.log_metric(f"{k}_mean", m)
            exp.log_metric(f"{k}_std", s)
            exp.log_metric(f"{k}_median", med)
        exp.log_metric("KL_real_pred", kl_val)
        exp.log_metric("JS_real_pred", js_val)
        exp.end()


if __name__ == "__main__":
    main()
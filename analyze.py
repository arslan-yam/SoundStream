import argparse
import random
from pathlib import Path
import comet_ml
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torchaudio
from src.model import SoundStreamModel

n_fft, hop, n_mels, max_sec = 1024, 256, 80, 15.0
SR = 16000
mel_fn = torchaudio.transforms.MelSpectrogram(SR, n_fft, hop_length=hop, n_mels=n_mels, power=2.0)
to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

def load_audio(path):
    data, sr = sf.read(str(path), dtype="float32")
    wav = torch.from_numpy(data)
    if wav.ndim > 1:
        wav = wav.mean(-1)
    if sr != SR:
        wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, SR).squeeze(0)
    return wav[: int(max_sec * SR)]


def stft_db(x):
    s = torch.stft(x, n_fft, hop, window=torch.hann_window(n_fft), return_complex=True)
    return 20 * torch.log10(s.abs().clamp_min(1e-6))


def make_figure(real, pred, title):
    specs = [
        ("waveform", real.numpy(), pred.numpy()),
        ("STFT (dB)", stft_db(real).numpy(), stft_db(pred).numpy()),
        ("mel (dB)", to_db(mel_fn(real)).numpy(), to_db(mel_fn(pred)).numpy()),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    for row, (name, r, p) in enumerate(specs):
        for col, data, label in [(0, r, "Real"), (1, p, "Predicted")]:
            ax = axes[row, col]
            if name == "waveform":
                ax.plot(np.arange(len(data)) / SR, data)
                ax.set_xlabel("time, s")
            else:
                ax.imshow(data, origin="lower", aspect="auto",
                          vmin=-80, vmax=max(r.max(), p.max()))
            ax.set_title(f"{label} {name}")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--n_audio", type=int, default=5)
    ap.add_argument("--checkpoint", default="saved/ss_5090_run1/model_best.pth")
    ap.add_argument("--workspace", default="arslan-yamaletdinov")
    ap.add_argument("--project", default="dl-big-hw")
    ap.add_argument("--run_name", default="ss_analysis")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SoundStreamModel().to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=False)
    files = sorted(str(p) for p in Path(args.data_dir).rglob("*") if p.suffix.lower() in {".flac", ".wav", ".mp3", ".ogg"})
    
    random.seed(42)
    random.shuffle(files)
    files = files[: args.n_audio]
    exp = comet_ml.Experiment(workspace=args.workspace, project_name=args.project, log_code=False, log_graph=False, auto_metric_logging=False, auto_param_logging=False)
    exp.set_name(args.run_name)
    exp.log_parameters(vars(args))

    for i, path in enumerate(files):
        real = load_audio(path)
        with torch.no_grad():
            pred = model(audio=real[None, None].to(device))["audio_pred"]
        pred = pred.squeeze().cpu().clamp(-1, 1)
        exp.log_audio(real.numpy(), SR, file_name=f"real_{i:02d}.wav", step=i)
        exp.log_audio(pred.numpy(), SR, file_name=f"pred_{i:02d}.wav", step=i)
        fig = make_figure(real, pred, title=Path(path).name)
        exp.log_figure(figure_name=f"analysis_{i:02d}", figure=fig, step=i)
        plt.close(fig)

    exp.end()

if __name__ == "__main__":
    main()
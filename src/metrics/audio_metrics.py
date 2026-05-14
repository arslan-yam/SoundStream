import torch
from torchmetrics.audio import NonIntrusiveSpeechQualityAssessment, ShortTimeObjectiveIntelligibility
from src.metrics.base_metric import BaseMetric


class STOIMetric(BaseMetric):
    def __init__(self, sample_rate=16000, device="auto", *args, **kwargs):
        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metric = ShortTimeObjectiveIntelligibility(fs=sample_rate).to(device)

    def __call__(self, audio, audio_pred, audio_lengths=None, **kwargs):
        scores = []
        for i in range(audio.shape[0]):
            t = audio_lengths[i].item() if audio_lengths is not None else audio.shape[-1]
            a = audio[i, 0, :t]
            p = audio_pred[i, 0, :t]
            scores.append(self.metric(p, a))
        return torch.stack(scores).mean()


class NISQAMetric(BaseMetric):
    def __init__(self, sample_rate=16000, device="auto", *args, **kwargs):
        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.metric = NonIntrusiveSpeechQualityAssessment(fs=sample_rate).to(device)

    def __call__(self, audio_pred, audio_lengths=None, **kwargs):
        scores = []
        for i in range(audio_pred.shape[0]):
            t = (audio_lengths[i].item() if audio_lengths is not None else audio_pred.shape[-1])
            p = audio_pred[i, 0, :t]
            out = self.metric(p)
            scores.append(out[0])
        return torch.stack(scores).mean()


class PerplexityMetric(BaseMetric):
    def __init__(self, codebook_size=1024, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.codebook_size = codebook_size

    def __call__(self, indices, **kwargs):
        counts = torch.bincount(indices.flatten(), minlength=self.codebook_size).float()
        probs = counts / counts.sum().clamp_min(1)
        probs = probs[probs > 0]
        ent = -(probs * probs.log()).sum()
        return ent.exp()

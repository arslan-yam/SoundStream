from pathlib import Path
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset


class LJSpeechDataset(Dataset):
    def __init__(self, root, tokenizer, sample_rate=16000, max_text_tokens=128, max_audio_sec=10.0):
        self.root = Path(root)
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.max_text_tokens = max_text_tokens
        self.max_audio_sec = max_audio_sec
        self.items = []
        for line in (self.root / "metadata.csv").read_text().splitlines():
            parts = line.split("|")
            if len(parts) >= 3:
                self.items.append((parts[0], parts[2] or parts[1]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        wav_id, text = self.items[i]
        audio, sr = sf.read(str(self.root / "wavs" / f"{wav_id}.wav"), dtype="float32")
        audio = torch.from_numpy(audio)
        if audio.ndim > 1:
            audio = audio.mean(-1)
        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio.unsqueeze(0), sr, self.sample_rate).squeeze(0)
        audio = audio[: int(self.max_audio_sec * self.sample_rate)]
        ids = self.tokenizer.encode(text, max_length=self.max_text_tokens, truncation=True, add_special_tokens=False)
        return {"audio": audio, "text_ids": torch.tensor(ids, dtype=torch.long)}


def tts_collate(batch):
    max_t = max(len(b["text_ids"]) for b in batch)
    max_a = max(b["audio"].shape[-1] for b in batch)
    text_ids = torch.zeros(len(batch), max_t, dtype=torch.long)
    audio = torch.zeros(len(batch), max_a)
    for i, b in enumerate(batch):
        text_ids[i, : len(b["text_ids"])] = b["text_ids"]
        audio[i, : b["audio"].shape[-1]] = b["audio"]
    return {"text_ids": text_ids, "audio": audio}
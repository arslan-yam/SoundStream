import random
from pathlib import Path
import torch
import torchaudio
from torch.utils.data import Dataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json


class LibriSpeechDataset(Dataset):
    def __init__(self, data_dir, part="train-clean-100", sample_rate=16000, crop_length=8000, limit=None):
        self.data_dir = Path(data_dir)
        self.part = part
        self.sample_rate = sample_rate
        self.crop_length = crop_length

        index_path = ROOT_PATH / "data" / "librispeech" / f"{part}.json"
        if index_path.exists():
            self._index = read_json(str(index_path))
        else:
            self._index = self._create_index()
            index_path.parent.mkdir(parents=True, exist_ok=True)
            write_json(self._index, str(index_path))

        if limit is not None:
            self._index = self._index[:limit]

    def _create_index(self):
        base = self.data_dir / self.part
        if not base.exists():
            base = self.data_dir
        files = sorted(str(p) for p in base.rglob("*.flac"))
        if not files:
            files = sorted(str(p) for p in base.rglob("*.wav"))
        return [{"path": p} for p in files]

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        path = self._index[idx]["path"]
        audio, sr = torchaudio.load(path)
        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sr, self.sample_rate)
        audio = audio.mean(dim=0)

        if self.crop_length is not None:
            t = audio.shape[0]
            if t < self.crop_length:
                n_repeat = self.crop_length // t + 1
                audio = audio.repeat(n_repeat)[: self.crop_length]
            else:
                start = random.randint(0, t - self.crop_length)
                audio = audio[start : start + self.crop_length]

        return {"audio": audio.unsqueeze(0)}

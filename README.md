# SoundStream Neural Audio Codec

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#data">Data</a> •
  <a href="#training">Training</a> •
  <a href="#inference--evaluation">Inference & Evaluation</a> •
  <a href="#demo">Demo</a> •
  <a href="#bonus-codec-based-tts">Bonus: TTS</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

<p align="center">
<a href="https://github.com/Blinorot/pytorch_project_template">
  <img src="https://img.shields.io/badge/built%20on-pytorch--template-green">
</a>
<img src="https://img.shields.io/badge/license-MIT-blue.svg">
</p>

## About

PyTorch implementation of [SoundStream: An End-to-End Neural Audio Codec](https://arxiv.org/abs/2107.03312) for 16 kHz speech, trained on `train-clean-100` of [LibriSpeech](https://www.openslr.org/12).

- **Encoder** strides `[2, 4, 5, 5]` — total downsample 200, latent rate 80 Hz at 16 kHz.
- **Residual Vector Quantizer**: 8 quantizers × 1024 entries → **6.4 kbps**.
- **Discriminators**: 3-scale wave + STFT, hinge GAN + feature matching loss.
- **Reconstruction**: mel + commitment loss.

The repo is based on the [pytorch-project-template](https://github.com/Blinorot/pytorch_project_template).

It also contains:

- a **no-GAN model** (same model trained without adversarial / feature-matching loss),
- a **codec-based TTS** system (SmolLM2-135M + X-Codec, LoRA fine-tuning) trained on LJSpeech.

All metrics, audio samples, and curves are logged to [Comet ML](https://www.comet.com/)

## Installation

```bash
# 1. clone
git clone https://github.com/arslan-yam/SoundStream.git
cd SoundStream

# 2. install Python deps
pip install -r requirements.txt

# 3. ffmpeg + torchcodec (needed by torchaudio.load on torch>=2.7)
#    Linux:
apt-get install -y ffmpeg
pip install torchcodec

#    macOS:
brew install ffmpeg
pip install torchcodec

```

Python 3.11 or 3.12 recommended. For RTX 50xx (for instace I used RTX5090 for training), torch needs the `cu128` index:

```bash
pip install --force-reinstall torch torchaudio torchvision \
  --index-url https://download.pytorch.org/whl/cu128
```

## Data

### LibriSpeech (for the codec)

```bash
mkdir -p data/librispeech && cd data/librispeech
wget https://www.openslr.org/resources/12/train-clean-100.tar.gz
wget https://www.openslr.org/resources/12/test-clean.tar.gz
tar -xzf train-clean-100.tar.gz && rm train-clean-100.tar.gz
tar -xzf test-clean.tar.gz && rm test-clean.tar.gz
cd ../..
```

### LJSpeech (for the TTS)

```bash
mkdir -p data/ljspeech && cd data/ljspeech
curl -L -o LJSpeech-1.1.tar.bz2 \
  "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
tar -xjf LJSpeech-1.1.tar.bz2 && rm LJSpeech-1.1.tar.bz2
cd ../..
```

## Pretrained checkpoints

Both checkpoints live on Google Drive. Download via `gdown`:

```python
import gdown, os
os.makedirs("saved/ss_final", exist_ok=True)
os.makedirs("saved/tts_lj_lora", exist_ok=True)

# Main SoundStream (GAN) checkpoint
gdown.download(id="1UchYZOlm3cDxykMwOOhPObgGnGKJdPu-", output="saved/ss_final/model_best.pth", quiet=False)

# TTS LoRA checkpoint
gdown.download(id="1XV94sTljhD_l76AYl57aT0UtVvwsF34A", output="saved/tts_lj_lora/model_best.pth", quiet=False)
```

(Run as a Python script or copy into a notebook cell.)

## Training

### SoundStream (main, with GAN)

```bash
export COMET_API_KEY=<your_key>
python3 train.py -cn=soundstream \
  datasets.train.data_dir=data/librispeech/LibriSpeech \
  datasets.test.data_dir=data/librispeech/LibriSpeech \
  trainer.n_epochs=20 trainer.epoch_len=3000 \
  writer.workspace=arslan-yamaletdinov \
  writer.project_name=dl-big-hw \
  writer.run_name=ss_main
```

Follows the SEANet/SoundStream recipe: Adam `(0.5, 0.9)`, constant LR `1e-4`, 0.5 s random crops, batch size 12, but I changes 450000 to 60 000 total steps (it takes approximately 2 hours on RTX 5090).

### Same model without GAN

```bash
python3 train.py -cn=soundstream \
  datasets.train.data_dir=data/librispeech/LibriSpeech \
  datasets.test.data_dir=data/librispeech/LibriSpeech \
  loss_function.lambda_adv=0.0 loss_function.lambda_feat=0.0 \
  trainer.n_epochs=20 trainer.epoch_len=3000 \
  writer.run_name=ss_no_gan
```

## Inference and Evaluation

Final metrics on the **full `test-clean`** (STOI / NISQA / Perplexity, full-length audio, batch 1):

```bash
python3 inference.py \
  datasets.test.data_dir=data/librispeech/LibriSpeech \
  inferencer.from_pretrained=saved/ss_final/model_best.pth \
  writer.run_name=ss_main_final_eval
```

Qualitative analysis (waveform + STFT + mel-spec side-by-side, audio uploaded to Comet) for a few files:

```bash
python3 analyze.py \
  --data_dir data/librispeech/LibriSpeech/test-clean \
  --n_audio 5 \
  --checkpoint saved/ss_final/model_best.pth \
  --run_name ss_analysis
```

Quantitative stats (SI-SDR, LSD, MCD, band energies, spectral centroid, KL and JS over waveform distributions etc.):

```bash
python3 stats.py \
  --data_dir data/librispeech/LibriSpeech/test-clean \
  --n_files 200 \
  --checkpoint saved/ss_final/model_best.pth \
  --comet --run_name ss_stats_test_clean
```

The same `analyze.py` / `stats.py` were run on a noisy english corpus (voicebank-demand) and russian speech (youtube meme clips via `yt-dlp`) — see the report for cross-corpus comparison.

## Demo

[`demo.ipynb`](demo.ipynb) — runs in Google Colab:

1. Clone repo, install deps.
2. Download `model_best.pth` from Drive via `gdown`.
3. Take a user-provided audio URL, run it through the codec, display original and re-synthesized audio.

Just open it in Colab and run the cells. The audio URL in Cell 5 can be replaced with any `.wav` / `.flac` / `.mp3`.

## Codec-based TTS

A small TTS system built on top of:

- **Frozen** [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) as the LM backbone.
- **Frozen** [X-Codec](https://huggingface.co/hf-audio/xcodec-hubert-librispeech) — only its **first** RVQ quantizer is used for audio tokens.
- **Trainable**: an audio token embedding, an LM head over audio vocabulary, and LoRA adapters (`r=8`, `q_proj/v_proj`) inside SmolLM2

Train:

```bash
python3 train_tts.py \
  --ljspeech_dir data/ljspeech/LJSpeech-1.1 \
  --n_epochs 30 --batch_size 8 \
  --use_lora --lora_r 8 --lora_alpha 16 \
  --save_dir saved/tts_lj_lora \
  --run_name tts_lj_lora
```

Every epoch the script generates audio for 4 held-out LJSpeech prompts and logs `val_STOI`, `val_NISQA`, plus pairs of `real / pred` `.wav` to Comet for listening.

Inference on a custom prompt:

```bash
python3 inference_tts.py \
  --checkpoint saved/tts_lj_lora/model_best.pth \
  --prompt "Hello world, this is my codec TTS model." \
  --use_lora --lora_r 8 --lora_alpha 16 \
  --out_path tts_output.wav \
  --comet --run_name tts_demo
```

[`demo_tts.ipynb`](demo_tts.ipynb) is the Colab-ready demo for the TTS path.

## Reports

The Comet project [`arslan-yamaletdinov/dl-big-hw`](https://www.comet.com/arslan-yamaletdinov/dl-big-hw/view/) collects:

- training curves of the main GAN model, the no-GAN ablation, and TTS,
- final STOI / NISQA on full test-clean,
- analysis runs on `test-clean`, noisy english (voicebank-demand), and Russian (youtube),
- audio samples for every analysis run.

## Credits

Project skeleton: [pytorch-project-template](https://github.com/Blinorot/pytorch_project_template).
Codec follows the architecture from the [SoundStream paper](https://arxiv.org/abs/2107.03312);
training recipe follows the [SEANet](https://arxiv.org/abs/2009.02095) setup.

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

import argparse
from pathlib import Path
import comet_ml
import torch
from torch.utils.data import DataLoader, Subset
from torchmetrics.audio import NonIntrusiveSpeechQualityAssessment, ShortTimeObjectiveIntelligibility
from tqdm import tqdm
from transformers import AutoTokenizer

from src.datasets.ljspeech import LJSpeechDataset, tts_collate
from src.model.tts_model import TTSModel


def evaluate(model, val_data, tokenizer, device, stoi, nisqa, exp, epoch):
    model.eval()
    stoi_vals, nisqa_vals = [], []
    for i, item in enumerate(val_data):
        text, real = item["text"], item["audio"]
        ids = tokenizer.encode(text, add_special_tokens=False)
        text_ids = torch.tensor(ids, device=device).unsqueeze(0)
        with torch.no_grad():
            codes = model.generate(text_ids, max_tokens=500, temperature=0.8)
            pred = model.decode_audio(codes).squeeze().cpu().clamp(-1, 1)
        p = pred[:min(len(pred), len(real))].to(device)
        r = real[:min(len(pred), len(real))].to(device)
        try:
            stoi_vals.append(stoi(p, r).item())
        except Exception:
            pass
        try:
            nisqa_vals.append(nisqa(p)[0].item())
        except Exception:
            pass

        exp.log_text(text, metadata={"name": f"val_text_{i}_epoch{epoch}"})
        exp.log_audio(r.cpu().numpy(), sample_rate=16000, file_name=f"real_{i}_epoch{epoch}.wav", step=epoch)
        exp.log_audio(pred.cpu().numpy(), sample_rate=16000, file_name=f"pred_{i}_epoch{epoch}.wav", step=epoch)

    if stoi_vals:
        exp.log_metric("val_STOI", sum(stoi_vals) / len(stoi_vals), step=epoch)
    if nisqa_vals:
        exp.log_metric("val_NISQA", sum(nisqa_vals) / len(nisqa_vals), step=epoch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ljspeech_dir", required=True)
    ap.add_argument("--lm_name", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--codec_name", default="hf-audio/xcodec-hubert-librispeech")
    ap.add_argument("--n_epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--save_dir", default="saved/tts")
    ap.add_argument("--n_val", type=int, default=4)
    ap.add_argument("--workspace", default="arslan-yamaletdinov")
    ap.add_argument("--project", default="dl-big-hw")
    ap.add_argument("--run_name", default="tts_lj")
    ap.add_argument("--use_lora", action="store_true")
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.lm_name)
    model = TTSModel(args.lm_name, args.codec_name, use_lora=args.use_lora, lora_r=args.lora_r, lora_alpha=args.lora_alpha).to(device)
    full_ds = LJSpeechDataset(args.ljspeech_dir, tokenizer)
    n_train = len(full_ds) - args.n_val
    train_ds = Subset(full_ds, range(n_train))
    val_data = []
    
    for i in range(args.n_val):
        idx = n_train + i
        val_data.append({"text": full_ds.items[idx][1], "audio": full_ds[idx]["audio"]})

    dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=tts_collate, num_workers=args.num_workers)
    stoi = ShortTimeObjectiveIntelligibility(fs=16000).to(device)
    nisqa = NonIntrusiveSpeechQualityAssessment(fs=16000).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    
    exp = comet_ml.Experiment(
        workspace=args.workspace, project_name=args.project,
        log_code=False, log_graph=False,
        auto_metric_logging=False, auto_param_logging=False,
    )
    exp.set_name(args.run_name)
    exp.log_parameters(vars(args))

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    step = 0
    for epoch in range(args.n_epochs):
        model.train()
        model.codec.eval()
        model.lm.eval()
        for batch in tqdm(dl, desc=f"epoch {epoch}"):
            audio = batch["audio"].to(device)
            text_ids = batch["text_ids"].to(device)
            with torch.no_grad():
                audio_codes = model.encode_audio(audio)
            loss = model(text_ids, audio_codes)["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            exp.log_metric("loss", loss.item(), step=step)
            step += 1

        evaluate(model, val_data, tokenizer, device, stoi, nisqa, exp, epoch)
        trainable_keys = {n for n, p in model.named_parameters() if p.requires_grad}
        trained = {k: v for k, v in model.state_dict().items() if k in trainable_keys}
        torch.save({"state_dict": trained, "args": vars(args)}, save_dir / f"model_epoch_{epoch}.pth")
    exp.end()


if __name__ == "__main__":
    main()
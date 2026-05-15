import argparse

import comet_ml
import soundfile as sf
import torch
from transformers import AutoTokenizer

from src.model.tts_model import TTSModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--lm_name", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--codec_name", default="hf-audio/xcodec-hubert-librispeech")
    ap.add_argument("--out_path", default="tts_output.wav")
    ap.add_argument("--max_tokens", type=int, default=500)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--comet", action="store_true")
    ap.add_argument("--workspace", default="arslan-yamaletdinov")
    ap.add_argument("--project", default="dl-big-hw")
    ap.add_argument("--run_name", default="tts_lj_infer")
    ap.add_argument("--use_lora", action="store_true")
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.lm_name)
    model = TTSModel(args.lm_name, args.codec_name, use_lora=args.use_lora, lora_r=args.lora_r, lora_alpha=args.lora_alpha).to(device).eval()
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    text_ids = torch.tensor(ids, device=device).unsqueeze(0)
    codes = model.generate(text_ids, max_tokens=args.max_tokens, temperature=args.temperature)
    audio = model.decode_audio(codes).squeeze().cpu().clamp(-1, 1).numpy()
    sf.write(args.out_path, audio, 16000)

    if args.comet:
        exp = comet_ml.Experiment(
            workspace=args.workspace, project_name=args.project,
            log_code=False, log_graph=False,
            auto_metric_logging=False, auto_param_logging=False,
        )
        exp.set_name(args.run_name)
        exp.log_parameters(vars(args))
        exp.log_text(args.prompt, metadata={"name": "prompt"})
        exp.log_audio(audio, sample_rate=16000, file_name="generated.wav")
        exp.log_metric("duration_sec", len(audio) / 16000)
        exp.log_metric("n_audio_tokens", codes.shape[-1])
        exp.end()


if __name__ == "__main__":
    main()
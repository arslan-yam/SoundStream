import torch
import torch.nn.functional
from torch import nn
from transformers import AutoModel
from peft import LoraConfig, get_peft_model


class TTSModel(nn.Module):
    def __init__(self, lm_name="HuggingFaceTB/SmolLM2-135M", codec_name="hf-audio/xcodec-hubert-librispeech", codebook_size=1024, 
                 use_lora=False, lora_r=8, lora_alpha=16):
        super().__init__()
        
        self.lm = AutoModel.from_pretrained(lm_name, torch_dtype=torch.float32)
        self.codec = AutoModel.from_pretrained(codec_name, trust_remote_code=True, torch_dtype=torch.float32)
        if use_lora:
            lora_cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05, target_modules=["q_proj", "v_proj"], bias="none")
            self.lm = get_peft_model(self.lm, lora_cfg)
        else:
            for p in self.lm.parameters():
                p.requires_grad = False
        for p in self.codec.parameters():
            p.requires_grad = False
        self.codec.eval()
        
        self.codebook_size = codebook_size
        self.bos_id = codebook_size
        self.eos_id = codebook_size + 1
        self.audio_vocab = codebook_size + 2
        hidden = self.lm.config.hidden_size
        self.audio_embed = nn.Embedding(self.audio_vocab, hidden)
        self.audio_head = nn.Linear(hidden, self.audio_vocab)

    @torch.no_grad()
    def encode_audio(self, audio):
        out = self.codec.encode(audio.unsqueeze(1))
        codes = out.audio_codes if hasattr(out, "audio_codes") else out["audio_codes"]
        if codes.dim() == 3 and codes.shape[0] != audio.shape[0]:
            codes = codes.transpose(0, 1)
        return codes[:, 0]

    @torch.no_grad()
    def decode_audio(self, codes_q1):
        n_q = getattr(self.codec.config, "num_quantizers", 8)
        codes = torch.zeros(codes_q1.shape[0], n_q, codes_q1.shape[1], dtype=torch.long, device=codes_q1.device,)
        codes[:, 0] = codes_q1
        audio = self.codec.decode(codes)
        audio = audio.audio_values if hasattr(audio, "audio_values") else audio
        return audio.squeeze(1) if audio.dim() == 3 else audio

    def forward(self, text_ids, audio_codes, **kw):
        B, T_text = text_ids.shape
        bos = torch.full((B, 1), self.bos_id, dtype=torch.long, device=text_ids.device)
        eos = torch.full((B, 1), self.eos_id, dtype=torch.long, device=text_ids.device)
        audio_in = torch.cat([bos, audio_codes], dim=1)
        audio_tgt = torch.cat([audio_codes, eos], dim=1)
        text_emb = self.lm.get_input_embeddings()(text_ids)
        audio_emb = self.audio_embed(audio_in)
        h = self.lm(inputs_embeds=torch.cat([text_emb, audio_emb], dim=1)).last_hidden_state
        logits = self.audio_head(h[:, T_text:])
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, self.audio_vocab), audio_tgt.reshape(-1))
        return {"loss": loss, "logits": logits}

    @torch.no_grad()
    def generate(self, text_ids, max_tokens=500, temperature=0.8, top_k=50):
        self.eval()
        B = text_ids.shape[0]
        text_emb = self.lm.get_input_embeddings()(text_ids)
        cur = torch.full((B, 1), self.bos_id, dtype=torch.long, device=text_ids.device)
        for _ in range(max_tokens):
            audio_emb = self.audio_embed(cur)
            h = self.lm(inputs_embeds=torch.cat([text_emb, audio_emb], dim=1)).last_hidden_state
            logits = self.audio_head(h[:, -1]) / temperature
            if top_k:
                v, idx = logits.topk(top_k, dim=-1)
                masked = torch.full_like(logits, float("-inf"))
                masked.scatter_(-1, idx, v)
                logits = masked
            nxt = torch.multinomial(torch.nn.functional.softmax(logits, dim=-1), 1)
            cur = torch.cat([cur, nxt], dim=1)
            if (nxt == self.eos_id).all():
                break
        codes = cur[:, 1:]
        codes = torch.where(codes < self.codebook_size, codes, torch.zeros_like(codes))
        return codes
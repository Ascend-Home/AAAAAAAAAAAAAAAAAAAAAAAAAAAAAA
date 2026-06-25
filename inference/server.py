"""FastAPI inference server."""
import os
import torch
import yaml
from fastapi import FastAPI
from pydantic import BaseModel
from model import MaxTransformer, MaxConfig
import sentencepiece as spm

app = FastAPI()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT = os.environ.get("CHECKPOINT_PATH", "checkpoints/dpo_final.pt")
TOKENIZER  = os.environ.get("TOKENIZER_PATH", "tokenizer/out/tokenizer.model")

cfg   = MaxConfig(**yaml.safe_load(open("config/max_config.yaml"))["model"])
model = MaxTransformer(cfg).to(DEVICE).eval()
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
sp    = spm.SentencePieceProcessor(model_file=TOKENIZER)


class Req(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate")
@torch.no_grad()
def gen(r: Req):
    ids    = torch.tensor([sp.encode(r.prompt)]).to(DEVICE)
    eos_id = sp.piece_to_id("<|im_end|>")
    out    = model.generate(
        ids,
        max_new_tokens=r.max_tokens,
        temperature=r.temperature,
        eos_token_id=eos_id,
    )
    return {"text": sp.decode(out[0].tolist())}

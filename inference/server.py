"""FastAPI inference server."""
import os, torch, yaml
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

@app.post("/generate")
@torch.no_grad()
def gen(r: Req):
    ids = torch.tensor([sp.encode(r.prompt)]).to(DEVICE)
    for _ in range(r.max_tokens):
        logits, _ = model(ids)
        logits = logits[:, -1] / r.temperature
        probs  = torch.softmax(logits, -1)
        nxt    = torch.multinomial(probs, 1)
        ids    = torch.cat([ids, nxt], 1)
        if nxt.item() == sp.piece_to_id("<|im_end|>"): break
    return {"text": sp.decode(ids[0].tolist())}

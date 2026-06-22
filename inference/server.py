"""FastAPI inference server."""
import torch, yaml
from fastapi import FastAPI
from pydantic import BaseModel
from model import MaxTransformer, MaxConfig
import sentencepiece as spm

app = FastAPI()
cfg = MaxConfig(**yaml.safe_load(open("config/max_config.yaml"))["model"])
model = MaxTransformer(cfg).cuda().eval()
model.load_state_dict(torch.load("/checkpoints/dpo_final.pt"))
sp = spm.SentencePieceProcessor(model_file="tokenizer/out/tokenizer.model")

class Req(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7

@app.post("/generate")
@torch.no_grad()
def gen(r: Req):
    ids = torch.tensor([sp.encode(r.prompt)]).cuda()
    for _ in range(r.max_tokens):
        logits, _ = model(ids)
        logits = logits[:, -1] / r.temperature
        probs = torch.softmax(logits, -1)
        nxt = torch.multinomial(probs, 1)
        ids = torch.cat([ids, nxt], 1)
        if nxt.item() == sp.piece_to_id("<|im_end|>"): break
    return {"text": sp.decode(ids[0].tolist())}

# run: uvicorn inference.server:app --host 0.0.0.0 --port 8000

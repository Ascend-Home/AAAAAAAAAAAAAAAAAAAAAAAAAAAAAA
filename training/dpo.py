"""Direct Preference Optimization."""
import torch, torch.nn.functional as F, json, os
from torch.utils.data import Dataset, DataLoader
from model import MaxTransformer, MaxConfig
import yaml, sentencepiece as spm
import copy

class DPODataset(Dataset):
    def __init__(self, path, tok, max_len=4096):
        self.sp = spm.SentencePieceProcessor(model_file=tok)
        self.data = [json.loads(l) for l in open(path)]
        self.max_len = max_len
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        d = self.data[i]
        prompt = self.sp.encode(d["prompt"])
        chosen = self.sp.encode(d["chosen"])
        rejected = self.sp.encode(d["rejected"])
        return (torch.tensor(prompt+chosen)[:self.max_len],
                torch.tensor(prompt+rejected)[:self.max_len],
                len(prompt))

def logprobs(model, ids, prompt_len):
    logits, _ = model(ids.unsqueeze(0))
    lp = F.log_softmax(logits[0, :-1], dim=-1)
    tgt = ids[1:]
    chosen_lp = lp.gather(1, tgt.unsqueeze(1)).squeeze()
    return chosen_lp[prompt_len-1:].sum()

def dpo_loss(pol_c, pol_r, ref_c, ref_r, beta=0.1):
    return -F.logsigmoid(beta * ((pol_c - ref_c) - (pol_r - ref_r))).mean()

def main():
    rank = 0; torch.cuda.set_device(rank)
    cfg = MaxConfig(**yaml.safe_load(open("config/max_config.yaml"))["model"])
    policy = MaxTransformer(cfg).to(rank)
    policy.load_state_dict(torch.load("/checkpoints/sft_final.pt"))
    ref = copy.deepcopy(policy); ref.eval()
    for p in ref.parameters(): p.requires_grad = False
    optim = torch.optim.AdamW(policy.parameters(), lr=1e-7)
    ds = DPODataset("/data/preferences.jsonl", "tokenizer/out/tokenizer.model")
    loader = DataLoader(ds, batch_size=1)
    for c, r, pl in loader:
        c, r = c.to(rank), r.to(rank)
        pc = logprobs(policy, c, pl); pr = logprobs(policy, r, pl)
        with torch.no_grad():
            rc = logprobs(ref, c, pl); rr = logprobs(ref, r, pl)
        loss = dpo_loss(pc, pr, rc, rr)
        loss.backward(); optim.step(); optim.zero_grad()
        print(f"dpo {loss.item():.4f}")
    torch.save(policy.state_dict(), "/checkpoints/dpo_final.pt")

if __name__ == "__main__": main()

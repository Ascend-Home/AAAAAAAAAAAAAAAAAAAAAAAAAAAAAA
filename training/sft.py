"""Supervised fine-tuning on instruction data."""
import torch, json, yaml, os
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from model import MaxTransformer, MaxConfig
import sentencepiece as spm

class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer_path, max_len=4096):
        self.sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
        self.data = [json.loads(l) for l in open(jsonl_path)]
        self.max_len = max_len
        self.im_start = self.sp.piece_to_id("<|im_start|>")
        self.im_end = self.sp.piece_to_id("<|im_end|>")
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        ex = self.data[i]
        ids, labels = [], []
        for turn in ex["messages"]:
            role_ids = self.sp.encode(turn["role"] + "\n")
            content_ids = self.sp.encode(turn["content"])
            seg = [self.im_start] + role_ids + content_ids + [self.im_end]
            ids.extend(seg)
            if turn["role"] == "assistant":
                labels.extend(seg)
            else:
                labels.extend([-100]*len(seg))
        ids = ids[:self.max_len]; labels = labels[:self.max_len]
        ids += [0]*(self.max_len - len(ids))
        labels += [-100]*(self.max_len - len(labels))
        return torch.tensor(ids), torch.tensor(labels)

def main():
    dist.init_process_group("nccl")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    cfg = MaxConfig(**yaml.safe_load(open("config/max_config.yaml"))["model"])
    model = MaxTransformer(cfg).to(rank)
    model.load_state_dict(torch.load("/checkpoints/pretrain_final.pt"))
    optim = torch.optim.AdamW(model.parameters(), lr=5e-6)
    ds = SFTDataset("/data/sft.jsonl", "tokenizer/out/tokenizer.model")
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    for ep in range(3):
        for x, y in loader:
            x, y = x.to(rank), y.to(rank)
            _, loss = model(x, y)
            loss.backward(); optim.step(); optim.zero_grad()
            if rank == 0: print(f"sft loss {loss.item():.4f}")
    if rank == 0: torch.save(model.state_dict(), "/checkpoints/sft_final.pt")

if __name__ == "__main__": main()

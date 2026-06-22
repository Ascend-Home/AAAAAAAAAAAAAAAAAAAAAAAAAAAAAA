"""Run MMLU, GPQA, HumanEval, etc."""
import torch, json, argparse, yaml
from model import MaxTransformer, MaxConfig
import sentencepiece as spm

BENCHMARKS = ["mmlu", "gpqa", "humaneval", "gsm8k", "math", "swebench"]

def eval_mmlu(model, sp, path):
    correct = 0; total = 0
    for line in open(path):
        ex = json.loads(line)
        prompt = ex["question"] + "\n" + "\n".join(f"{c}. {a}" for c,a in zip("ABCD", ex["choices"])) + "\nAnswer:"
        ids = torch.tensor([sp.encode(prompt)]).cuda()
        with torch.no_grad():
            logits, _ = model(ids)
        pred_id = logits[0, -1].argmax().item()
        pred = sp.decode([pred_id]).strip()
        if pred == ex["answer"]: correct += 1
        total += 1
    return correct / total

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--bench", choices=BENCHMARKS, default="mmlu")
    args = p.parse_args()
    cfg = MaxConfig(**yaml.safe_load(open("config/max_config.yaml"))["model"])
    m = MaxTransformer(cfg).cuda(); m.load_state_dict(torch.load(args.ckpt)); m.eval()
    sp = spm.SentencePieceProcessor(model_file="tokenizer/out/tokenizer.model")
    acc = eval_mmlu(m, sp, f"/data/eval/{args.bench}.jsonl")
    print(f"{args.bench}: {acc*100:.2f}%")

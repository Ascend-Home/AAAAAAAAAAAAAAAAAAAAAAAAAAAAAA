"""End-to-end: raw text → tokenized binary shards."""
import argparse, glob, os, numpy as np
import sentencepiece as spm
from multiprocessing import Pool
from tqdm import tqdm

def tokenize_file(args):
    path, tok_path, out_dir = args
    sp = spm.SentencePieceProcessor(model_file=tok_path)
    out_path = os.path.join(out_dir, os.path.basename(path) + ".bin")
    with open(path) as f:
        text = f.read()
    ids = sp.encode(text)
    arr = np.array(ids, dtype=np.uint32)
    arr.tofile(out_path)
    return len(ids)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--workers", type=int, default=64)
    args = p.parse_args()
    os.makedirs(args.output, exist_ok=True)
    files = sorted(glob.glob(args.input))
    jobs = [(f, args.tokenizer, args.output) for f in files]
    total = 0
    with Pool(args.workers) as pool:
        for n in tqdm(pool.imap_unordered(tokenize_file, jobs), total=len(jobs)):
            total += n
    print(f"Tokenized {total/1e9:.2f}B tokens")

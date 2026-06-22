"""MinHash LSH deduplication."""
import argparse, glob, os, hashlib
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm

def shingle(text, k=5):
    return {text[i:i+k] for i in range(len(text)-k+1)}

def dedup(input_glob, output_dir, threshold=0.8):
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    os.makedirs(output_dir, exist_ok=True)
    seen = 0
    for path in tqdm(sorted(glob.glob(input_glob))):
        with open(path) as f, open(os.path.join(output_dir, os.path.basename(path)), "w") as out:
            for i, doc in enumerate(f.read().split("<|endoftext|>")):
                if len(doc) < 200: continue
                m = MinHash(num_perm=128)
                for s in shingle(doc): m.update(s.encode())
                key = f"{path}:{i}"
                if not lsh.query(m):
                    lsh.insert(key, m)
                    out.write(doc + "\n<|endoftext|>\n")
                    seen += 1
    print(f"Kept {seen} unique docs")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    dedup(args.input, args.output)

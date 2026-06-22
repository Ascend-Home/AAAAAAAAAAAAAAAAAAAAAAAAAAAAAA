"""Download and stream Common Crawl WARC files."""
import requests, gzip, io, argparse, os
from concurrent.futures import ThreadPoolExecutor
from warcio.archiveiterator import ArchiveIterator

CC_INDEX = "https://data.commoncrawl.org/crawl-data/CC-MAIN-{snapshot}/warc.paths.gz"

def get_warc_paths(snapshot="2024-46"):
    r = requests.get(CC_INDEX.format(snapshot=snapshot))
    return gzip.decompress(r.content).decode().splitlines()

def process_warc(path, output_dir):
    url = f"https://data.commoncrawl.org/{path}"
    out = os.path.join(output_dir, os.path.basename(path).replace(".warc.gz", ".txt"))
    with requests.get(url, stream=True) as r, open(out, "w") as f:
        for record in ArchiveIterator(r.raw):
            if record.rec_type == "response":
                try:
                    text = record.content_stream().read().decode("utf-8", errors="ignore")
                    f.write(text + "\n<|endoftext|>\n")
                except: pass

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", default="2024-46")
    p.add_argument("--output", default="/data/cc_raw")
    p.add_argument("--workers", type=int, default=64)
    args = p.parse_args()
    os.makedirs(args.output, exist_ok=True)
    paths = get_warc_paths(args.snapshot)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(lambda p: process_warc(p, args.output), paths))

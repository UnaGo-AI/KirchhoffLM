"""Build eval_texts.json: 40 Wikipedia (WikiText-103 validation) and 40 Python source excerpts of 6,000
characters each, the held-out texts used for every bits-per-byte number in this project."""
import glob, json, os, random, re

import pyarrow.parquet as pq
import transformers as tf
from huggingface_hub import hf_hub_download, list_repo_files

ROOT = os.path.dirname(os.path.abspath(__file__))
CHARS, N = 6000, 40
ARTICLE = re.compile(r"^ = [^=].* = $")                      # top-level heading starts a new article


def articles(split):
    files = sorted(f for f in list_repo_files("Salesforce/wikitext", repo_type="dataset")
                   if f.startswith(f"wikitext-103-raw-v1/{split}-") and f.endswith(".parquet"))
    if not files:
        raise RuntimeError(f"no parquet files for split {split}")
    lines = []
    for f in files:
        lines += pq.read_table(hf_hub_download("Salesforce/wikitext", f, repo_type="dataset")).column("text").to_pylist()
    docs, cur = [], []
    for line in lines:
        if ARTICLE.match(line.rstrip("\n")) and cur:
            docs.append("".join(cur))
            cur = []
        cur.append(line)
    if cur:
        docs.append("".join(cur))
    return docs


def main():
    rng = random.Random(1234)
    wiki = [a for a in articles("validation") if len(a) > CHARS + 200]
    files = sorted(glob.glob(os.path.join(os.path.dirname(tf.__file__), "**", "*.py"), recursive=True))
    code = [t for t in (open(f, encoding="utf-8", errors="ignore").read() for f in files[:400]) if len(t) > CHARS + 200]

    def cut(texts):
        out = []
        for _ in range(N):
            t = rng.choice(texts)
            s = rng.randrange(0, len(t) - CHARS)
            out.append(t[s:s + CHARS])
        return out
    json.dump({"wikipedia": cut(wiki), "code": cut(code)}, open(os.path.join(ROOT, "eval_texts.json"), "w"))
    print("wrote eval_texts.json")


if __name__ == "__main__":
    main()

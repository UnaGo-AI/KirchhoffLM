"""Download one TinyStories shard, train a small BPE tokenizer, write uint16 token files."""
import os
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

DATA = os.environ.get("KLM_TINY_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "tinystories"))
VOCAB = 4096
EOT = "<|eot|>"


def fetch(split_prefix: str) -> str:
    files = [f for f in list_repo_files("roneneldan/TinyStories", repo_type="dataset")
             if f.startswith(f"data/{split_prefix}-") and f.endswith(".parquet")]
    if not files:
        raise RuntimeError(f"no parquet shard found for split {split_prefix}")
    return hf_hub_download("roneneldan/TinyStories", sorted(files)[0], repo_type="dataset", local_dir=DATA)


def main():
    os.makedirs(DATA, exist_ok=True)
    train_texts = pq.read_table(fetch("train")).column("text").to_pylist()
    val_texts = pq.read_table(fetch("validation")).column("text").to_pylist()
    print(f"train stories {len(train_texts):,}  val stories {len(val_texts):,}")

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=VOCAB, special_tokens=[EOT],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(train_texts[:200_000], trainer)
    tok.save(os.path.join(DATA, "tokenizer.json"))
    eot = tok.token_to_id(EOT)

    for name, texts in (("train", train_texts), ("val", val_texts)):
        ids = []
        for enc in tok.encode_batch(texts):
            ids.extend(enc.ids)
            ids.append(eot)
        arr = np.array(ids, dtype=np.uint16)
        arr.tofile(os.path.join(DATA, f"{name}.bin"))
        print(f"{name}: {len(arr):,} tokens")


if __name__ == "__main__":
    main()

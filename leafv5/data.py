"""Data pipeline for LEAFv5 training.

Sources (streamed, byte-capped -> no need to download whole datasets):
  * --data shakespeare   : Tiny Shakespeare (char LM, zero deps)
  * --data tinystories   : TinyStories (HF mirror, plain-text stream)
  * --data wikitext      : WikiText-103 raw (HF mirror, plain-text stream)
  * --data-file PATH     : any local UTF-8 text file

Tokenizers:
  * char  (default for shakespeare / --data-file)  -- no dependencies
  * bpe   (default for tinystories / wikitext)     -- trained with `tokenizers`
        (fast Rust), then *encoded with GigaToken* (Rust, GB/s, exact HF parity)
        when available:  --tokenizer-engine gigatoken | hf | auto

The tokenized corpus is written once to a uint16 memmap (data_dir/corpus.bin)
with meta.json, then batches are sliced directly from the mmap.  A background
prefetcher (BatchPrefetcher) overlaps CPU batch assembly with GPU training.
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
import urllib.request
from typing import Dict, Iterator, List, Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------
# URLs
# ---------------------------------------------------------------------------
SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
TINYSTORIES_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt"
WIKITEXT_URL = ("https://huggingface.co/datasets/Salesforce/wikitext/resolve/master/"
                "wikitext-103-raw/wiki.train.raw")


def gigatoken_available() -> bool:
    try:
        import gigatoken  # noqa: F401
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Streaming text iterators (yield str chunks)
# ---------------------------------------------------------------------------
def _iter_bytes_url(url: str, max_bytes: Optional[int]) -> Iterator[str]:
    req = urllib.request.Request(url, headers={"User-Agent": "leafv5-slm/0.1"})
    with urllib.request.urlopen(req, timeout=120) as r:
        total = 0
        while True:
            chunk = r.read(1 << 20)  # 1 MiB
            if not chunk:
                break
            total += len(chunk)
            yield chunk.decode("utf-8", errors="ignore")
            if max_bytes is not None and total >= max_bytes:
                break


def _iter_file(path: str, max_bytes: Optional[int]) -> Iterator[str]:
    total = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            total += len(chunk)
            yield chunk
            if max_bytes is not None and total >= max_bytes:
                break


def iter_source(source: str, data_file: Optional[str], max_bytes: Optional[int]) -> Iterator[str]:
    if source == "shakespeare":
        cache = os.path.join("data_cache", "tinyshakespeare.txt")
        if not os.path.exists(cache):
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            print(f"[data] downloading Tiny Shakespeare -> {cache}")
            req = urllib.request.Request(SHAKESPEARE_URL, headers={"User-Agent": "leafv5-slm/0.1"})
            with urllib.request.urlopen(req, timeout=120) as r, open(cache, "wb") as f:
                f.write(r.read())
        yield from _iter_file(cache, max_bytes)
    elif source == "tinystories":
        print(f"[data] streaming TinyStories from HF mirror (cap={max_bytes} bytes)")
        yield from _iter_bytes_url(TINYSTORIES_URL, max_bytes)
    elif source == "wikitext":
        print(f"[data] streaming WikiText-103 raw from HF mirror (cap={max_bytes} bytes)")
        yield from _iter_bytes_url(WIKITEXT_URL, max_bytes)
    elif source == "file":
        assert data_file, "--data-file required when --data file"
        yield from _iter_file(data_file, max_bytes)
    else:
        raise ValueError(f"unknown source {source!r}")


# ---------------------------------------------------------------------------
# Tokenizers
# ---------------------------------------------------------------------------
class CharTokenizer:
    mode = "char"

    def __init__(self, vocab: Dict[str, int]):
        self.vocab = vocab
        self.itos = {i: c for c, i in vocab.items()}

    def encode(self, text: str) -> List[int]:
        return [self.vocab[c] for c in text]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @classmethod
    def train(cls, text_iter: Iterator[str], sample_bytes: int = 2_000_000) -> "CharTokenizer":
        chars = set()
        n = 0
        for chunk in text_iter:
            chars.update(chunk)
            n += len(chunk)
            if n >= sample_bytes:
                break
        chars = sorted(chars)
        return cls({c: i for i, c in enumerate(chars)})


class BPETokenizer:
    """Byte-level BPE.  Trained with HuggingFace `tokenizers` (exact, proven);
    encoding may be done with GigaToken (Rust, GB/s, exact parity) when engine
    is 'gigatoken' -- falls back to HF automatically.
    """

    mode = "bpe"

    def __init__(self, bpe=None, engine: str = "auto", vocab_size: int = 0):
        self.bpe = bpe  # HF tokenizer or GigaToken Tokenizer
        self.engine = engine
        self._vocab_size = vocab_size or getattr(bpe, "vocab_size", None) or 0

    def encode(self, text: str) -> List[int]:
        ids = self.bpe.encode(text)
        if hasattr(ids, "ids"):  # HuggingFace tokenizers returns an Encoding
            ids = ids.ids
        return list(ids)

    def decode(self, ids) -> str:
        out = self.bpe.decode(list(ids))
        if isinstance(out, bytes):  # GigaToken returns bytes for byte-level BPE
            out = out.decode("utf-8", errors="replace")
        return out

    @property
    def vocab_size(self) -> int:
        if self._vocab_size:
            return self._vocab_size
        return self.bpe.get_vocab_size()

    @classmethod
    def train(cls, text_iter: Iterator[str], vocab_size: int = 16384,
              sample_bytes: int = 20_000_000) -> "BPETokenizer":
        from tokenizers import ByteLevelBPETokenizer
        tok = ByteLevelBPETokenizer()
        buff: List[str] = []
        n = 0
        for chunk in text_iter:
            buff.append(chunk)
            n += len(chunk)
            if n >= sample_bytes:
                break
        if not buff:
            raise ValueError("empty training sample for BPE (source yielded no text)")
        print(f"[data] training byte-level BPE (vocab={vocab_size}) on {n/1e6:.1f} MB...")
        tok.train_from_iterator(buff, vocab_size=vocab_size, min_frequency=2,
                                special_tokens=[])
        return cls(tok, engine="hf", vocab_size=tok.get_vocab_size())


def _gt_from_json(json_path: str, vocab_size: int):
    """Build a GigaToken-backed BPETokenizer from a saved HF tokenizer.json."""
    import gigatoken as gt
    gt_tok = gt.Tokenizer(json_path)
    return BPETokenizer(gt_tok, engine="gigatoken", vocab_size=vocab_size)


def save_tokenizer(tok, dirpath: str) -> Dict:
    os.makedirs(dirpath, exist_ok=True)
    if tok.mode == "char":
        with open(os.path.join(dirpath, "char_vocab.json"), "w") as f:
            json.dump(tok.vocab, f)
        return {"mode": "char", "dir": os.path.abspath(dirpath), "vocab_size": tok.vocab_size}
    # BPE: always save HF vocab/merges + tokenizer.json (portable, reloadable)
    tok.bpe.save(os.path.join(dirpath, "tokenizer.json"), pretty=True)
    os.makedirs(os.path.join(dirpath, "model"), exist_ok=True)
    tok.bpe.save_model(os.path.join(dirpath, "model"))
    return {"mode": "bpe", "dir": os.path.abspath(dirpath),
            "vocab_size": tok.vocab_size, "engine": tok.engine}


def load_tokenizer(meta: Dict, engine: str = "auto"):
    d = meta["tokenizer"]
    if d["mode"] == "char":
        with open(os.path.join(d["dir"], "char_vocab.json")) as f:
            return CharTokenizer(json.load(f))
    # BPE: prefer GigaToken (fast) when requested/available, else HF
    use_gt = (engine == "auto" and gigatoken_available()) or engine == "gigatoken"
    if use_gt:
        try:
            return _gt_from_json(os.path.join(d["dir"], "tokenizer.json"), d["vocab_size"])
        except Exception as e:
            print(f"[data] GigaToken load failed ({e}); falling back to HF tokenizers")
    from tokenizers import ByteLevelBPETokenizer
    return BPETokenizer(ByteLevelBPETokenizer.from_file(
        os.path.join(d["dir"], "model", "vocab.json"),
        os.path.join(d["dir"], "model", "merges.txt")),
        engine="hf", vocab_size=d["vocab_size"])


# ---------------------------------------------------------------------------
# Corpus preparation / loading
# ---------------------------------------------------------------------------
def prepare_corpus(source: str, data_file: Optional[str], tokenizer_mode: str,
                   vocab_size: int, data_dir: str, max_tokens: Optional[int],
                   val_frac: float = 0.02, max_val_tokens: int = 5_000_000,
                   tokenizer_engine: str = "auto", force: bool = False) -> Dict:
    """Tokenize the (capped) source into data_dir/corpus.bin + meta.json. Idempotent.

    BPE encoding uses GigaToken's native Rust file encoding when available
    (engine 'auto'/'gigatoken') for a ~50-1000x speedup over HF on large corpora.
    """
    data_dir = os.path.abspath(data_dir)
    os.makedirs(data_dir, exist_ok=True)
    meta_path = os.path.join(data_dir, "meta.json")
    if os.path.exists(meta_path) and not force:
        with open(meta_path) as f:
            return json.load(f)

    t0 = time.time()
    if tokenizer_mode == "auto":
        tokenizer_mode = "char" if source in ("shakespeare", "file") else "bpe"
    if tokenizer_mode == "char":
        tok = CharTokenizer.train(iter_source(source, data_file, None))
        engine = "none"
    else:
        if vocab_size > 65535:
            raise ValueError("BPE vocab_size must fit in uint16 ids (<= 65535)")
        tok = BPETokenizer.train(iter_source(source, data_file, None), vocab_size)
        engine = tokenizer_engine
        if engine == "auto":
            engine = "gigatoken" if gigatoken_available() else "hf"
    print(f"[data] tokenizer ready: {tok.mode}, vocab={tok.vocab_size}, "
          f"encode-engine={engine} ({time.time()-t0:.0f}s)")

    # ---- tokenize + write uint16 corpus ----
    bin_path = os.path.join(data_dir, "corpus.bin")
    n_tokens = 0
    raw_dir = os.path.join(data_dir, "raw_parts")
    os.makedirs(raw_dir, exist_ok=True)

    if tokenizer_mode == "char":
        n_tokens = _tokenize_streaming(tok, source, data_file, max_tokens, bin_path)
    elif engine == "gigatoken":
        n_tokens = _tokenize_gigatoken(tok, source, data_file, max_tokens,
                                       bin_path, raw_dir)
    else:
        n_tokens = _tokenize_streaming(tok, source, data_file, max_tokens, bin_path)

    n_val = min(max_val_tokens, int(n_tokens * val_frac))
    tok_meta = save_tokenizer(tok, os.path.join(data_dir, "tokenizer"))
    meta = {
        "source": source,
        "tokenizer": tok_meta,
        "vocab_size": tok.vocab_size,
        "n_tokens": int(n_tokens),
        "n_train": int(n_tokens - n_val),
        "n_val": int(n_val),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[data] corpus cached: {n_tokens/1e6:.1f}M tokens -> {bin_path} "
          f"({time.time()-t0:.0f}s)")
    return meta


def _encode_linewise(tok, text: str) -> List[int]:
    """Encode text one COMPLETE line at a time.

    P0 fix (#8): byte-level BPE merges can span arbitrary chunk boundaries, so
    encoding 1-MB chunks independently can differ from encoding the full text.
    Byte-level pretokenizers split on whitespace, so a token NEVER spans a
    newline -> encoding whole lines is exactly equivalent to encoding the
    full text.  Char tokenizers are per-char and unaffected."""
    if tok.mode == "char":
        return tok.encode(text)
    ids: List[int] = []
    for line in text.splitlines(keepends=True):
        ids.extend(tok.encode(line))
    return ids


def _tokenize_streaming(tok, source, data_file, max_tokens, bin_path):
    """Generic streaming encode (char tokenizer or HF BPE fallback), always
    on complete lines so streaming == full-text tokenization exactly."""
    n_tokens = 0
    ids_buf: List[int] = []
    line_buf: List[str] = []
    with open(bin_path, "wb") as f:
        for chunk in iter_source(source, data_file, max_tokens and max_tokens * 4):
            line_buf.append(chunk)
            # only encode complete lines (keep the trailing partial line buffered)
            joined = "".join(line_buf)
            if "\n" not in joined:
                continue
            last_nl = joined.rfind("\n")
            complete, rest = joined[:last_nl + 1], joined[last_nl + 1:]
            ids_buf.extend(_encode_linewise(tok, complete))
            line_buf = [rest]
            if len(ids_buf) >= 4_000_000:
                np.asarray(ids_buf, dtype=np.uint16).tofile(f)
                n_tokens += len(ids_buf)
                ids_buf = []
                if max_tokens and n_tokens >= max_tokens:
                    break
        if line_buf and line_buf[0]:
            ids_buf.extend(_encode_linewise(tok, line_buf[0]))
        if ids_buf:
            np.asarray(ids_buf, dtype=np.uint16).tofile(f)
            n_tokens += len(ids_buf)
    return n_tokens


def _tokenize_gigatoken(tok, source, data_file, max_tokens, bin_path, raw_dir):
    """GigaToken native encode: stream raw text into capped part files, then
    Rust-encode each part at GB/s and append uint16 ids."""
    import gigatoken as gt
    n_tokens = 0
    done = False
    part_no = 0
    raw_buf: List[str] = []
    raw_bytes = 0
    PART_BYTES = 16 << 20  # 16 MB text per part (~20-25M tokens)
    max_bytes = max_tokens and max_tokens * 4
    gtok = gt.Tokenizer(tok.bpe)  # wraps HF tokenizer, exact parity, GB/s native encode
    with open(bin_path, "wb") as f:
        for chunk in iter_source(source, data_file, max_bytes):
            if done:
                break
            raw_buf.append(chunk)
            raw_bytes += len(chunk)
            if raw_bytes < PART_BYTES:
                continue
            part = os.path.join(raw_dir, f"part-{part_no:05d}.txt")
            with open(part, "w", encoding="utf-8") as pf:
                pf.write("".join(raw_buf))
            raw_buf, raw_bytes = [], 0
            ids = np.asarray(gtok.encode_files(gt.TextFileSource([part]))[0],
                             dtype=np.uint16)
            os.remove(part)
            part_no += 1
            if max_tokens and n_tokens + ids.size > max_tokens:
                ids = ids[:max_tokens - n_tokens]
                done = True
            ids.tofile(f)
            n_tokens += ids.size
            print(f"[data] gigatoken encoded part {part_no} "
                  f"({n_tokens/1e6:.1f}M tokens)")
        if raw_buf and not done:
            part = os.path.join(raw_dir, f"part-{part_no:05d}.txt")
            with open(part, "w", encoding="utf-8") as pf:
                pf.write("".join(raw_buf))
            ids = np.asarray(gtok.encode_files(gt.TextFileSource([part]))[0],
                             dtype=np.uint16)
            os.remove(part)
            part_no += 1
            if max_tokens and n_tokens + ids.size > max_tokens:
                ids = ids[:max_tokens - n_tokens]
            ids.tofile(f)
            n_tokens += ids.size
            print(f"[data] gigatoken encoded final part {part_no} "
                  f"({n_tokens/1e6:.1f}M tokens)")
    return n_tokens


class Corpus:
    """Random-access uint16 token stream over the memmap."""

    def __init__(self, meta: Dict, data_dir: str):
        self.meta = meta
        self.arr = np.memmap(os.path.join(os.path.abspath(data_dir), "corpus.bin"),
                             dtype=np.uint16, mode="r")
        self.tokenizer = load_tokenizer(meta)

    @property
    def n_tokens(self) -> int:
        return self.meta["n_tokens"]

    def sample_batch(self, bs: int, seq: int, rng: np.random.Generator,
                     split: str = "train"):
        """Returns (x, y) torch long tensors of shape [bs, seq].

        seq is clamped to the largest window that fits the corpus
        (o + seq + 1 <= n_tokens must hold for every offset, so the y slice
        never runs past the end -- bug fix 2026-08-09: on corpora smaller than
        a window the old code produced x/y of different lengths and crashed)."""
        n_tok = self.meta["n_tokens"]
        if n_tok < 2:
            raise ValueError("corpus too small to sample (n_tokens < 2)")
        seq = min(seq, n_tok - 1)
        if split == "train":
            lo = self.meta["n_val"]
            hi = n_tok - seq - 1
            if hi <= lo:
                lo, hi = 0, max(0, n_tok - seq - 1)
            offsets = rng.integers(lo, max(hi, lo + 1), size=bs)
        else:
            # val region = last n_val tokens: [n_tokens - n_val, n_tokens).
            lo = max(0, n_tok - self.meta["n_val"])
            hi = n_tok - seq - 1
            if hi <= lo:
                lo, hi = 0, max(0, n_tok - seq - 1)
            n_b = max(1, (hi - lo) // seq)
            offsets = lo + (np.arange(bs) % n_b) * seq
        xs = np.stack([self.arr[o:o + seq] for o in offsets])
        ys = np.stack([self.arr[o + 1:o + seq + 1] for o in offsets])
        return (torch.from_numpy(xs.astype(np.int64)),
                torch.from_numpy(ys.astype(np.int64)))

    def text_from_ids(self, ids) -> str:
        return self.tokenizer.decode([int(i) for i in ids])


class StreamCorpus:
    """Contiguous token streams for state-carry (truncated-BPTT) training/eval.

    Each of `bs` streams advances through the corpus by `seq` tokens per call,
    so consecutive windows are contiguous and the recurrent state can be carried
    across them.  Streams wrap at the split boundary.  `sample_batch` has the
    same signature as `Corpus.sample_batch` so StreamCorpus plugs into
    `BatchPrefetcher` unchanged.
    """

    def __init__(self, corpus: Corpus, bs: int, seq: int, split: str = "train",
                 seed: int = 0):
        self.corpus = corpus
        self.bs = bs
        self.seq = seq
        self.split = split
        self.arr = corpus.arr
        meta = corpus.meta
        if split == "train":
            self.lo = meta["n_val"]
            self.hi = meta["n_tokens"] - seq - 1
        else:
            self.lo = 0
            self.hi = meta["n_val"] - seq - 1
        if self.hi <= self.lo:
            self.lo, self.hi = 0, max(0, corpus.n_tokens - seq - 2)
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.offsets = self.rng.integers(self.lo, max(self.hi, self.lo + 1), size=self.bs)

    def sample_batch(self, bs: int, seq: int, rng=None, split: str = "train"):
        # clamp so o + seq + 1 never runs past the end of the array
        # (bug fix 2026-08-09: on tiny corpora the y slice was truncated,
        # mismatching x and crashing downstream cross_entropy)
        seq = min(seq, max(1, int(self.corpus.n_tokens) - 1))
        xs, ys = [], []
        for i in range(bs):
            j = i % self.bs
            o = int(self.offsets[j])
            if o >= self.hi or o < self.lo or o + seq + 1 > self.corpus.n_tokens:
                o = int(self.rng.integers(self.lo, max(self.hi, self.lo + 1)))
            xs.append(self.arr[o:o + seq])
            ys.append(self.arr[o + 1:o + seq + 1])
            self.offsets[j] = o + seq
        return (torch.from_numpy(np.stack(xs).astype(np.int64)),
                torch.from_numpy(np.stack(ys).astype(np.int64)))


class BatchPrefetcher:
    """Background-thread batch assembler: overlaps CPU memmap reads + numpy
    slicing with GPU compute.  Pass batches with .get() in the training loop."""

    def __init__(self, corpus: Corpus, bs: int, seq: int, split: str = "train",
                 buffer: int = 4, seed: int = 0):
        self.corpus = corpus
        self.bs, self.seq, self.split = bs, seq, split
        self.q: "queue.Queue" = queue.Queue(maxsize=buffer)
        self.rng = np.random.default_rng(seed)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._produce, daemon=True)
        self._thread.start()

    def _produce(self):
        while not self._stop.is_set():
            x, y = self.corpus.sample_batch(self.bs, self.seq, self.rng, self.split)
            self.q.put((x, y))

    def get(self):
        return self.q.get()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)

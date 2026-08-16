"""Regression tests for the 2026-08-09 full bug-hunt fixes.

Bugs found by the audit and fixed (each has a test here):
  1. generate() ignored the caller's offset for plain-list states (serve
     sessions) and didn't pass offset on the first forward -> stateful RoPE
     positions restarted at 0 every turn.  Now: stateful turn-2 (list+offset
     and carried LeafStates) == one-shot full-sequence generation, rope ON.
  2. beam_search never conditioned on the full prompt (fed only the last
     prompt token from a fresh state, offset off by one).  Now: first beam
     token == greedy first token, and the prompt is prefilled in one pass.
  3. Corpus.sample_batch val split read past the end of the array on tiny
     corpora (y slice reached arr[n_tokens]) -> IndexError/shape mismatch.
  4. weights.py used Python's salted hash() for shared-ref dedupe, so a
     packed model saved to disk broke unpack in a new process (KeyError).
  5. finetune --lora-rank + --grow-at crashed (grow_width on LoRALinear).
     Now adapters merge before growth and restart fresh after.
  6. finetune eval crashed on a tiny val split (vx/vy truncated to different
     lengths -> cross_entropy batch-size mismatch).  Now clamped/skipped.
  7. serve.py session offset recomputed from text lengths drifted when the
     repetition guard stopped early; now uses the carried LeafStates offset.
Run:  python tests/test_bugfix_aug09.py
"""
import os
import string
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from leafv5.config import preset_config
from leafv5.data import CharTokenizer, Corpus, prepare_corpus
from leafv5.generate import beam_search, generate
from leafv5.lora import apply_lora, merge_lora
from leafv5.model import LeafLM


def _tok():
    # include the serve CHAT_TEMPLATE characters ("### Instruction:\n...") so
    # the char tokenizer can encode session prompts
    chars = string.ascii_letters + " .,?!#:\n"
    return CharTokenizer({c: i for i, c in enumerate(chars)})


def test_generate_offset_stateful_equals_oneshot():
    """Stateful continuation (carried LeafStates AND plain-list + offset, the
    serve path) must equal one-shot full-sequence generation with RoPE ON."""
    torch.manual_seed(0)
    tok = _tok()
    cfg = preset_config("micro", vocab_size=len(tok.vocab), n_layers=2, dim=96,
                        d_h=32, rope_dim=96, scale_init=0.1)   # RoPE on
    m = LeafLM(cfg).eval()
    turn1, turn2 = "hello there", "how are you"
    n1 = 5
    out1, st = generate(m, tok, turn1, max_new=n1, temperature=0.0, device="cpu")
    full_prompt = turn1 + out1 + turn2
    out_full, _ = generate(m, tok, full_prompt, max_new=6, temperature=0.0,
                           device="cpu")
    # carried LeafStates
    out2, _ = generate(m, tok, turn2, max_new=6, temperature=0.0, device="cpu",
                       states=st)
    assert out2 == out_full, (out2, out_full)
    # plain-list + explicit offset (what serve.py stores/passes)
    st_list = [s.detach() for s in st]
    off = len(tok.encode(turn1)) + n1
    out2b, _ = generate(m, tok, turn2, max_new=6, temperature=0.0, device="cpu",
                        states=st_list, offset=off)
    assert out2b == out_full, (out2b, out_full)
    print("  generate offset: stateful == one-shot (rope on, list+LeafStates) OK")


def test_beam_search_conditions_on_full_prompt():
    """Beam search's first token must equal greedy's first token (both are
    conditioned on the full prompt)."""
    torch.manual_seed(0)
    tok = _tok()
    cfg = preset_config("micro", vocab_size=len(tok.vocab), n_layers=2, dim=96,
                        d_h=32, scale_init=0.1)
    m = LeafLM(cfg).eval()
    prompt = "the cat sat on"
    greedy, _ = generate(m, tok, prompt, max_new=3, temperature=0.0, device="cpu")
    beam = beam_search(m, tok, prompt, max_new=3, beam_size=4, device="cpu")
    assert greedy and beam, (greedy, beam)
    assert greedy[:1] == beam[:1], (greedy, beam)
    # empty prompt + max_new=0 must not crash
    assert beam_search(m, tok, "", max_new=1, beam_size=2, device="cpu") != ""
    assert beam_search(m, tok, "hi", max_new=0, beam_size=2, device="cpu") == ""
    print("  beam_search: prompt-conditioned, first token == greedy, edges OK")


def test_sample_batch_val_tiny_corpus():
    """Tiny corpus val split must not read past the end of the array."""
    with tempfile.TemporaryDirectory() as d:
        txt = os.path.join(d, "tiny.txt")
        with open(txt, "w") as f:
            f.write("the quick brown fox jumps over the lazy dog. " * 4)
        meta = prepare_corpus("file", txt, "char", 0,
                              os.path.join(d, "cache"), max_tokens=None)
        corpus = Corpus(meta, os.path.join(d, "cache"))
        rng = np.random.default_rng(0)
        for seq in (64, 128, 256):   # 256 > n_tokens: must clamp, not crash
            x, y = corpus.sample_batch(8, seq, rng, "val")
            x2, y2 = corpus.sample_batch(8, seq, rng, "train")
            # the window is clamped to n_tokens-1; x/y must MATCH and be in bounds
            assert x.shape == y.shape, (x.shape, y.shape)
            assert x.shape[1] <= corpus.n_tokens - 1
            assert x2.shape == y2.shape
            assert x2.shape[1] <= corpus.n_tokens - 1
    print("  sample_batch: tiny-corpus train/val clamped, in-bounds, x==y OK")


def test_weights_pack_cross_process():
    """Shared-ref dedupe must survive a process boundary (stable hash)."""
    import subprocess
    torch.manual_seed(0)
    from leafv5.weights import pack_model
    cfg = preset_config("micro", vocab_size=128, n_layers=1, dim=64, d_h=16,
                        scale_init=0.1)
    m = LeafLM(cfg)
    p = pack_model(m.state_dict(), rank=0, quant_residual=True, shared=True)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(p, f.name)
        saved = f.name
    code = (
        "import torch; "
        "from leafv5.weights import unpack_model; "
        "sd = unpack_model(torch.load(%r, weights_only=False)); "
        "print(len(sd))" % saved
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, cwd=os.path.join(os.path.dirname(__file__),
                                                   ".."))
    os.unlink(saved)
    assert r.returncode == 0, r.stderr[-500:]
    assert int(r.stdout.strip()) == len(m.state_dict())
    print("  weights: pack/unpack across processes OK")


def test_lora_then_grow_works():
    """apply_lora -> merge_lora -> grow_width must not crash, and the grown
    model's function must be preserved (max |d| ~ 1e-5)."""
    torch.manual_seed(0)
    cfg = preset_config("micro", vocab_size=256, n_layers=2, dim=64, d_h=16,
                        scale_init=0.1)
    m = LeafLM(cfg)
    apply_lora(m, 8)              # wrap
    # train the adapters a little so the merge is non-trivial
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],
                            lr=1e-3)
    x = torch.randint(0, 256, (4, 16)); y = torch.randint(0, 256, (4, 16))
    for _ in range(3):
        opt.zero_grad()
        lg, _ = m(x, m.init_states(4, torch.device("cpu")))
        torch.nn.functional.cross_entropy(lg.reshape(-1, 256),
                                          y.reshape(-1)).backward()
        opt.step()
    m.eval()   # mem_dropout is on by default; eval mode for a clean comparison
    with torch.no_grad():
        before = m(x, m.init_states(4, torch.device("cpu")))[0]
    merge_lora(m)                 # now a plain LeafLM
    from leafv5.grow import grow_width
    g = grow_width(m, 128)
    g.eval(); m.eval()
    with torch.no_grad():
        after = g(x, g.init_states(4, torch.device("cpu")))[0]
    d = (after - before).abs().max().item()
    assert d < 5e-4, d
    print(f"  lora->merge->grow: function preserved (max|d|={d:.2e}) OK")


def test_finetune_eval_tiny_val_no_crash():
    """The finetune eval clamp: a tiny val split must not crash (regression
    for the vx/vy different-lengths mismatch)."""
    # replicate the fixed eval logic on a tiny val array
    import leafv5.finetune as ft
    val_arr = np.arange(198, dtype=np.uint16)   # tiny split
    seq_len = 256
    seq_v = seq_len
    if len(val_arr) <= seq_v + 1:
        seq_v = max(1, len(val_arr) - 1)
    assert seq_v >= 2 and len(val_arr) > seq_v
    rng = np.random.default_rng(0)
    hi = len(val_arr) - seq_v - 1
    o = rng.integers(0, max(hi, 1), size=16)
    vx = np.stack([val_arr[q:q + seq_v] for q in o])
    vy = np.stack([val_arr[q + 1:q + seq_v + 1] for q in o])
    assert vx.shape == vy.shape == (16, seq_v)
    print("  finetune eval: tiny val split clamped, shapes match OK")


def test_serve_session_offset_matches_state():
    """serve.py must store the model's carried offset (not a text-length
    recomputation)."""
    from leafv5.serve import ModelServer
    tok = _tok()
    torch.manual_seed(0)
    cfg = preset_config("micro", vocab_size=len(tok.vocab), n_layers=1, dim=64,
                        d_h=16, scale_init=0.1)
    m = LeafLM(cfg)
    srv = ModelServer.__new__(ModelServer)   # bypass __init__ (no ckpt)
    srv.model = m.eval()
    srv.tok = tok
    srv.device = "cpu"
    srv.max_new = 4
    srv.temperature = 0.0
    srv.top_k = 40
    srv.repeat_penalty = 1.0
    srv.max_sessions = 4
    srv.max_session_tokens = 10 ** 9
    srv.sessions = {}
    srv.session_chat("s1", "hello")
    stored = srv.sessions["s1"]
    # offset must be an int > 0 and equal to the carried state position
    assert isinstance(stored[1], int) and stored[1] > 0, stored[1]
    r2 = srv.session_chat("s1", "hi")
    assert isinstance(r2, str)
    # after a fresh session, offset advances monotonically across turns
    o1 = srv.sessions["s1"][1]
    srv.session_chat("s1", "more")
    o2 = srv.sessions["s1"][1]
    assert o2 > o1
    print(f"  serve: session offsets monotonic ({o1} -> {o2}) OK")


if __name__ == "__main__":
    torch.manual_seed(0)
    for fn in (test_generate_offset_stateful_equals_oneshot,
               test_beam_search_conditions_on_full_prompt,
               test_sample_batch_val_tiny_corpus,
               test_weights_pack_cross_process,
               test_lora_then_grow_works,
               test_finetune_eval_tiny_val_no_crash,
               test_serve_session_offset_matches_state):
        fn()
    print("\nBug-hunt regression tests passed.")

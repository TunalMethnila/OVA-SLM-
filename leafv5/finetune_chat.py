"""Interactive chat with a fine-tuned LEAFv5 checkpoint.

Usage:
    python -m leafv5.finetune_chat --ckpt out/leafv5-finetuned/best.pt
    # optional: --system "You are LEAFv5..." (only shown; the model already
    # learned its identity from the dataset)
"""
from __future__ import annotations

import argparse

from .generate import generate, load_checkpoint


def main():
    p = argparse.ArgumentParser(description="Chat with a fine-tuned LEAFv5.")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--repeat-penalty", type=float, default=1.3)
    p.add_argument("--max-new", type=int, default=120)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    model, tok, ck = load_checkpoint(args.ckpt, args.device)
    print("LEAFv5 chat (Ctrl-D to quit). Ask anything — including 'Who are you?'")

    template = "### Instruction:\n{instruction}\n\n### Response:\n"
    while True:
        try:
            q = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        out, _ = generate(model, tok, template.format(instruction=q),
                          max_new=args.max_new, temperature=args.temperature,
                          top_k=args.top_k, repeat_penalty=args.repeat_penalty,
                          device=args.device)
        print(f"LEAFv5: {out.strip()}")


if __name__ == "__main__":
    main()

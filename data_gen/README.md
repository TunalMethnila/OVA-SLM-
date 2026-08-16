# LEAFv5 Identity + Skills Dataset

A high-quality, seeded, synthetic instruction dataset that teaches LEAFv5:
**who it is** (LEAFv5, created by a single researcher, D.M.T.M.Dassanayake) and
a broad set of practical skills (reasoning, instruction following, tool use,
grammar, language, knowledge, creative writing, coding, safety).

## Files
- `make_dataset.py` — the (fully deterministic, seeded) generator. Edit it to
  extend any category, then re-run to regenerate.
- `leafv5_training_data.jsonl` — **23,735 examples**, one JSON object per line:
  `{"id", "category", "instruction", "output"}`.

## Contents (15 categories)

| category | examples | what it teaches |
|---|---|---|
| identity | 1,000 | the model knows it is LEAFv5, made by a **single researcher, D.M.T.M.Dassanayake** (40+ phrasings: who/what/created/built/made/how many people/architecture/credit/origin/mission) |
| reasoning_math | 3,200 | arithmetic with **guaranteed non-negative, verified answers** (56/56 spot-recompute check) |
| reasoning_word | 2,000 | shopping totals, ages, speed, percentages, equal sharing |
| reasoning_logic | 1,000 | syllogisms, modus ponens/tollens, ordering |
| reasoning_commonsense | 1,600 | curated "why" Q&A (why is the sky blue, why do we sleep...) |
| instruction | 2,400 | formal rephrasing, bullet lists, yes/no-then-explain, one-sentence answers |
| tools | 2,002 | function calling with a fixed toolset, JSON output, multi-step sequences |
| grammar | 4,000 | correction + explanation pairs (subject-verb, tense, articles, comparatives, apostrophes...) |
| language_sinhala | 1,600 | English ↔ Sinhala phrases + simple conversation (Sri Lanka context) |
| language_writing | 133 | writing styles (welcome, thank-you, invitation, apology, motivation) |
| knowledge | 2,000 | curated science / history / geography / Sri Lanka facts |
| creative | 1,600 | story prompts + haiku |
| coding | 1,200 | Python functions with solutions |
| safety | 600 | polite refusals for harmful/illegal/private requests |
| social | 800 | greetings & small talk (natural conversation warm-up) |

## Quality notes (honest)
- Math answers are computed by the generator and spot-verified; the regenerator
  enforces non-negative results.
- Knowledge / grammar / Sinhala entries are **curated banks** (written by hand
  in the generator) — no LLM-generated hallucination risk.
- Sinhala is a practical starter set (common phrases + transliteration); it is
  not exhaustive. Extend `SINHALA_BANK` in `make_dataset.py` to grow it.
- Some category sizes are template multiples (e.g. grammar emits 2 per base
  pair), so exact counts drift slightly from the target mix.

## Fine-tune with it

```bash
# full dataset (T4, ~1-3 h):
python -m leafv5.finetune --data data_gen/leafv5_training_data.jsonl \
    --model t4-4h --auto --steps 3000 --outdir out/leafv5-finetuned

# CPU smoke (proves the pipeline; a micro model will still repeat tokens):
python -m leafv5.finetune --data data_gen/leafv5_training_data.jsonl \
    --model micro --n-layers 2 --dim 160 --d-h 32 \
    --categories identity,reasoning_math,grammar,tools,language_sinhala \
    --max-samples 1200 --steps 400 --seq-len 128 --micro-batch 8

# identity-only (fastest way to see "who am I" learned):
python -m leafv5.finetune --data data_gen/leafv5_training_data.jsonl \
    --model micro --n-layers 2 --dim 160 --d-h 32 --categories identity \
    --max-samples 700 --steps 400 --seq-len 128 --micro-batch 8

# chat with the fine-tuned model:
python -m leafv5.finetune_chat --ckpt out/leafv5-finetuned/best.pt
```

The trainer formats each example as:
```
### Instruction:
{instruction}

### Response:
{output}
```
and fine-tunes with next-token cross-entropy (standard instruction tuning).

## Generation note (important bug fix)
`leafv5/generate.py` had an off-by-one: it re-fed the last prompt token as the
first generated token, double-writing it into the delta memory and corrupting
the state (fine-tuned models collapsed to newline loops). Fixed: the first
generated token is sampled from the prompt-pass logits, then only new tokens
are fed. Verified: a fine-tuned micro model answers "Who are you?" with
"LEAFv5 / D.M.T.M.Dassanayake / single researcher" content.

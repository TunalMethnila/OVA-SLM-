"""Serve a LEAFv5 model as a tiny HTTP API (stdlib only — no FastAPI/flask).

Endpoints:
  GET  /                -> model info (name, params, features)
  POST /generate        -> {"prompt": str, "max_new": int, "temperature": float,
                            "top_k": int, "repeat_penalty": float}
                            -> {"output": str}
  POST /chat            -> {"messages": [{"role": "user", "content": str}, ...],
                            ...same sampling opts}
                            -> {"output": str}

Usage:
  python -m leafv5.serve --ckpt out/leafv5-finetuned/best.pt --port 8000
  curl -s localhost:8000/generate -d '{"prompt":"Who are you?"}'
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

import torch

from .generate import generate, load_checkpoint

CHAT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"


def build_chat(messages) -> str:
    """Multi-turn chat -> the finetune template.  Every user turn becomes an
    instruction (the final one is what we complete); assistant replies are
    appended as context."""
    text = ""
    for i, m in enumerate(messages):
        if m.get("role") == "user":
            text += CHAT_TEMPLATE.format(instruction=m["content"])
        elif m.get("role") == "assistant":
            text += m["content"] + "\n\n"
    return text


class ModelServer:
    def generate(self, prompt: str, max_new: Optional[int] = None,
                 temperature: Optional[float] = None, top_k: Optional[int] = None,
                 repeat_penalty: Optional[float] = None,
                 states=None, offset: int = 0):
        """Generate; may carry a recurrent state (stateful sessions)."""
        return generate(
            self.model, self.tok, prompt,
            max_new=max_new or self.max_new,
            temperature=self.temperature if temperature is None else temperature,
            top_k=self.top_k if top_k is None else top_k,
            repeat_penalty=self.repeat_penalty if repeat_penalty is None
            else repeat_penalty,
            device=self.device, states=states, offset=offset)

    def info(self) -> dict:
        cfg = self.model.cfg
        return {
            "name": "LEAFv5",
            "creator": "D.M.T.M.Dassanayake (single researcher)",
            "params": self.model.n_params,
            "dim": cfg.dim, "layers": cfg.n_layers,
            "heads": f"{cfg.fast_heads}/{cfg.medium_heads}/{cfg.slow_heads}",
            "d_h": cfg.d_h, "vocab": self.tok.vocab_size,
            "sessions": len(self.sessions),
        }

    # ---- stateful sessions: the delta memory IS the conversation context ----
    # Each session keeps only the tiny recurrent state + position offset, so
    # multi-turn cost is CONSTANT (history is never re-encoded), unlike a
    # transformer's KV cache which grows per turn.
    def __init__(self, ckpt: str, device: str, temperature: float = 0.7,
                 top_k: int = 40, repeat_penalty: float = 1.3,
                 max_new: int = 120, max_session_tokens: int = 4096,
                 max_sessions: int = 32):
        self.model, self.tok, self.ck = load_checkpoint(ckpt, device)
        self.temperature = temperature
        self.top_k = top_k
        self.repeat_penalty = repeat_penalty
        self.max_new = max_new
        self.max_session_tokens = max_session_tokens
        self.max_sessions = max_sessions
        self.device = device if device != "auto" else \
            ("cuda" if torch.cuda.is_available() else "cpu")
        self.sessions: dict = {}  # session_id -> (states, offset, turns)
        print(f"[serve] LEAFv5 ready: {self.model.n_params/1e6:.1f}M params, "
              f"vocab {self.tok.vocab_size}, on {self.device}")

    def session_chat(self, session_id: str, user_msg: str,
                     max_new: Optional[int] = None,
                     temperature: Optional[float] = None,
                     top_k: Optional[int] = None,
                     repeat_penalty: Optional[float] = None) -> str:
        """Chat within a session: carry the recurrent state across turns.
        Returns the assistant reply; stores the updated state."""
        states, offset, turns = self.sessions.get(session_id,
                                                  (None, 0, 0))
        prompt = CHAT_TEMPLATE.format(instruction=user_msg)
        if len(self.sessions) >= self.max_sessions and session_id not in self.sessions:
            # evict the oldest session
            self.sessions.pop(next(iter(self.sessions)))
        out, new_states = generate(
            self.model, self.tok, prompt,
            max_new=max_new or self.max_new,
            temperature=self.temperature if temperature is None else temperature,
            top_k=self.top_k if top_k is None else top_k,
            repeat_penalty=self.repeat_penalty if repeat_penalty is None
            else repeat_penalty,
            device=self.device, states=states, offset=offset)
        turns += 1
        # Use the EXACT absolute position carried by the returned LeafStates
        # (bug fix 2026-08-09: recomputing from prompt+output lengths drifts
        # when the repetition guard stops generation before max_new -- the
        # offset then disagreed with the actual state position).
        offset = int(getattr(new_states, "offset", 0))
        # reset if the session exceeds the budget (bounded memory)
        if offset >= self.max_session_tokens:
            states, offset, turns = None, 0, 0
        else:
            states = [s.detach() for s in new_states]
        self.sessions[session_id] = (states, offset, turns)
        return out.strip()


def main():
    p = argparse.ArgumentParser(description="Serve LEAFv5 as an HTTP API.")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--device", default="auto")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--repeat-penalty", type=float, default=1.3)
    p.add_argument("--max-new", type=int, default=120)
    args = p.parse_args()

    server = ModelServer(args.ckpt, args.device, args.temperature, args.top_k,
                         args.repeat_penalty, args.max_new)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/info"):
                self._send(200, server.info())
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            try:
                n = int(self.headers.get("Content-Length", 0))
                data = json.loads(self.rfile.read(n) or b"{}")
            except Exception as e:
                self._send(400, {"error": f"bad request: {e}"})
                return
            try:
                if self.path == "/generate":
                    out = server.generate(
                        data.get("prompt", ""),
                        max_new=data.get("max_new"),
                        temperature=data.get("temperature"),
                        top_k=data.get("top_k"),
                        repeat_penalty=data.get("repeat_penalty"))
                    self._send(200, {"output": out})
                elif self.path == "/chat":
                    msgs = data.get("messages", [])
                    if not msgs or msgs[-1].get("role") != "user":
                        self._send(400, {"error": "last message must be a user turn"})
                        return
                    sid = data.get("session_id")
                    if sid:
                        # stateful: carry the recurrent state across turns
                        out = server.session_chat(
                            sid, msgs[-1]["content"],
                            **{k: data[k] for k in
                               ("max_new", "temperature", "top_k",
                                "repeat_penalty") if k in data})
                        self._send(200, {"output": out,
                                         "session_id": sid,
                                         "turns": server.sessions.get(
                                             sid, (None, 0, 0))[2]})
                        return
                    prompt = build_chat(msgs)
                    out = server.generate(prompt, **{k: data[k] for k in
                        ("max_new", "temperature", "top_k", "repeat_penalty")
                        if k in data})
                    self._send(200, {"output": out})
                else:
                    self._send(404, {"error": "not found"})
            except Exception as e:
                self._send(500, {"error": str(e)})

        def log_message(self, *a):
            pass  # quiet

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[serve] listening on http://{args.host}:{args.port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

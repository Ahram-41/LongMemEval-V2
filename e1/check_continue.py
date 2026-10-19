#!/usr/bin/env python3
"""Sanity check: does continue_final_message resume the Qwen3.5 reasoning inside <think>?
Prints the rendered prompt tail (via /tokenize + /detokenize) and the continuation."""
import json, os, urllib.request
base = os.environ["READER_BASE_URL"].rsplit("/v1", 1)[0]; model = os.environ.get("READER_MODEL", "Qwen/Qwen3.5-9B")
msgs = [{"role": "user", "content": "What is 17*23? Put the final answer in \\boxed{}."},
        {"role": "assistant", "content": "<think>\nLet me compute 17*23. 17*20 = 340, and 17*3 = 51, so"}]
def post(path, body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=300).read())
tok = post("/tokenize", {"model": model, "messages": msgs, "add_generation_prompt": False, "continue_final_message": True})
txt = post("/detokenize", {"model": model, "tokens": tok["tokens"]})["prompt"]
print("RENDERED TAIL:", repr(txt[-220:]))
r = post("/v1/chat/completions", {"model": model, "messages": msgs, "max_tokens": 300, "temperature": 0.6,
                                  "continue_final_message": True, "add_generation_prompt": False})
m = r["choices"][0]["message"]
print("REASONING:", repr((m.get("reasoning_content") or m.get("reasoning") or "")[:400]))
print("CONTENT:", repr((m.get("content") or "")[:400]))

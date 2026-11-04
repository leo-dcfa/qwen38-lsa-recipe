"""Prefill and decode speed at one context length, one request at a time.

Runs inside the image, against a server already up on this box. Appends one line per
request to results/bench.jsonl:
  prefill tok/s = prompt tokens / time to the first token
  decode tok/s  = tokens after the first / the time they took
The answer is greedy and ignores EOS, so every request decodes the same number of tokens.
"""
import argparse
import json
import os
import time
import urllib.request

from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--label", required=True)
parser.add_argument("--snapshot", required=True, help="the model snapshot, for its tokenizer")
parser.add_argument("--text", required=True, help="the pinned ShareGPT file")
parser.add_argument("--contexts", type=int, nargs="+", required=True)
parser.add_argument("--repeats", type=int, default=2)
parser.add_argument("--output-tokens", type=int, default=64)
parser.add_argument("--recipe-commit", default="")
parser.add_argument("--url", default="http://127.0.0.1:8000")
parser.add_argument("--results", default="/results/bench.jsonl")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.snapshot)
longest = max(args.contexts)
texts = []
chars = 0
# About 4 characters a token, with room to spare.
for row in json.load(open(args.text)):
    for turn in row["conversations"]:
        texts.append(turn["value"])
        chars += len(turn["value"])
    if chars > longest * 8:
        break
ids = tokenizer("\n\n".join(texts), add_special_tokens=False)["input_ids"]
if len(ids) < longest:
    raise SystemExit(f"bench.py: the text has {len(ids)} tokens, {longest} asked for")


def request(prompt):
    """Send one prompt; return the times of the first and last tokens and the final meta_info."""
    body = {
        "input_ids": prompt,
        "stream": True,
        "sampling_params": {"temperature": 0, "max_new_tokens": args.output_tokens, "ignore_eos": True},
    }
    sent = time.monotonic()
    req = urllib.request.Request(args.url + "/generate", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    first = last = None
    first_count = meta = None
    # Long enough for a 512K prefill on one GB10.
    with urllib.request.urlopen(req, timeout=4 * 3600) as answer:
        for line in answer:
            if not line.startswith(b"data:") or line.startswith(b"data: [DONE]"):
                continue
            meta = json.loads(line[5:])["meta_info"]
            now = time.monotonic()
            if meta.get("completion_tokens", 0) >= 1:
                if first is None:
                    first, first_count = now, meta["completion_tokens"]
                last = now
    if meta is None or first is None:
        raise SystemExit("bench.py: no tokens came back")
    return first - sent, last - first, first_count, meta


os.makedirs(os.path.dirname(args.results), exist_ok=True)
for context in args.contexts:
    for repeat in range(args.repeats):
        # Each repeat starts further into the text, so no two prompts are the same.
        offset = (len(ids) - context) * repeat // max(args.repeats, 1)
        ttft, decoding, first_count, meta = request(ids[offset:offset + context])
        decoded = meta["completion_tokens"] - first_count
        row = {
            "label": args.label,
            "recipe_commit": args.recipe_commit,
            "context": context,
            "repeat": repeat,
            "prompt_tokens": meta["prompt_tokens"],
            "cached_tokens": meta.get("cached_tokens", 0),
            "output_tokens": meta["completion_tokens"],
            "ttft_s": round(ttft, 2),
            "prefill_tok_s": round(context / ttft, 1),
            "decode_tok_s": round(decoded / decoding, 2) if decoding > 0 else None,
        }
        with open(args.results, "a") as sink:
            sink.write(json.dumps(row) + "\n")
        print("bench.py:", row, flush=True)

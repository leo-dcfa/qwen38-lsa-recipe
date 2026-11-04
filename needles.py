"""Does a server still find one fact hidden in a long text?

A check built on the same idea as RULER needle-in-a-haystack, written for this recipe and not
RULER itself, so its scores cannot be read against the paper. One sentence holding a 7-digit
number is planted at a random place in real text, and the model is asked for the number with
thinking off and greedy decoding. Everything is seeded, so both servers get the same prompts.
One line per question goes to results/needles.jsonl.
"""
import argparse
import json
import random
import time
import urllib.request

from transformers import AutoTokenizer

CITIES = ["Lisbon", "Nairobi", "Osaka", "Quito", "Tallinn", "Hobart", "Accra", "Bergen", "Cusco", "Dakar",
          "Fukuoka", "Gdansk", "Hanoi", "Izmir", "Jaipur", "Krakow", "Leipzig", "Medellin", "Nantes", "Porto"]

parser = argparse.ArgumentParser()
parser.add_argument("--label", required=True)
parser.add_argument("--snapshot", required=True)
parser.add_argument("--text", required=True)
parser.add_argument("--contexts", type=int, nargs="+", required=True)
parser.add_argument("--rows", type=int, default=10)
parser.add_argument("--recipe-commit", default="")
parser.add_argument("--url", default="http://127.0.0.1:8000")
parser.add_argument("--results", default="/results/needles.jsonl")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.snapshot)
longest = max(args.contexts)
texts = []
chars = 0
for row in json.load(open(args.text)):
    for turn in row["conversations"]:
        texts.append(turn["value"])
        chars += len(turn["value"])
    # About 4 characters a token, so this is about twice the longest context.
    if chars > longest * 8:
        break
ids = tokenizer("\n\n".join(texts), add_special_tokens=False)["input_ids"]
if len(ids) < longest:
    raise SystemExit(f"needles.py: the text has {len(ids)} tokens, {longest} asked for")

for context in args.contexts:
    for row in range(args.rows):
        rng = random.Random(f"{context}/{row}")
        city = rng.choice(CITIES)
        number = rng.randrange(1_000_000, 10_000_000)
        start = rng.randrange(0, len(ids) - context)
        hay = ids[start:start + context]
        # Past the first 4 keys and before the last 2,048, so a sparse read must find it.
        position = rng.randrange(1024, context - 2048)
        before = tokenizer.decode(hay[:position])
        after = tokenizer.decode(hay[position:])
        needle = f"\n\nThe special magic number for {city} is: {number}.\n\n"
        prompt = ("A special magic number is hidden in the text below. Remember it, because you will be "
                  f"asked for it afterwards.\n\n{before}{needle}{after}\n\n"
                  f"What is the special magic number for {city}? Answer with the number only.")
        body = {"model": "qwen3.8-27b", "temperature": 0, "max_tokens": 32,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": prompt}]}
        sent = time.monotonic()
        req = urllib.request.Request(args.url + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        answer = json.load(urllib.request.urlopen(req, timeout=4 * 3600))
        text = answer["choices"][0]["message"]["content"] or ""
        line = {
            "label": args.label, "recipe_commit": args.recipe_commit, "context": context, "row": row,
            "position": position, "expected": number, "answer": text.strip()[:64],
            "correct": str(number) in text, "prompt_tokens": answer["usage"]["prompt_tokens"],
            "seconds": round(time.monotonic() - sent, 1),
        }
        with open(args.results, "a") as sink:
            sink.write(json.dumps(line) + "\n")
        print("needles.py:", line, flush=True)

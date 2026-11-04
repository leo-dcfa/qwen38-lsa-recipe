# qwen38-lsa-recipe

A sparkrun recipe that serves **Qwen3.8-27B with Latent Sparse Attention (D-LSA)** on one DGX Spark (GB10, sm121), and measures whether it is faster than the same model with full attention.

This is an independent recipe, not affiliated with the LSA authors.

## What LSA is

[Latent Sparse Attention](https://arxiv.org/abs/2610.42147) (Kubicek et al., 2026) lets each query read a few keys instead of all of them, and finds those keys by searching clusters of keys in a learned space rather than scoring every key.
The distilled variant, D-LSA, adds small index projections to a frozen model; the weights of the model do not change.
The authors publish a map for this model in their [code release](https://github.com/Prefrontal-AI/latent-sparse-attention) (Apache-2.0): `qwen38_27b__long-16k-1M-heavy-key-and-query-d2048.pt`, release `d-lsa-weights-v1`.

## Why one GB10

- Qwen3.8-27B in BF16 is about 54 GB, more than a 32 GB card holds, and the map was distilled on the BF16 checkpoint.
- LSA makes finding keys cheaper but not storing them, so the KV cache still grows with the context.
  The model has 16 full-attention layers (of 64) with 4 KV heads of 256, which is 64 KiB of BF16 KV per token: 512K tokens is 32 GiB.
- A GB10's 128 GB of unified memory holds the weights and a KV cache of about 512K tokens.

## What the paper reports for this model

Per attention layer at 10M tokens, against full attention: 72.4x prefill and 42.5x decode; time to first token 9.2x faster.
The paper says the gains are smaller below 1M tokens.
Only the 16 full-attention layers can get faster; the 48 linear-attention layers cost the same either way.
One GB10 reaches about 512K, so the question this repo answers is: **is there a gain at 128K-512K on a GB10?**

## Plan

1. Pin every source: the LSA repo, the authors' SGLang fork, the model revision, the map and its sha256.
2. Read what the fork adds to SGLang.
3. Build an image for aarch64 + sm121 from the fork (as of 2026-11-03 the authors' serving image cannot be pulled anonymously).
4. Recipe for the baseline (plugin installed, not armed = full attention), dry run, launch.
5. Bench script: prefill and decode at 128K, 256K, 512K.
6. Arm LSA (one commit), measure again, and check that accuracy holds.

On GPUs other than Hopper, the plugin reads the selected keys through SGLang's Triton decode kernel, and its own kernels default to Triton, so FlashAttention-3 is not needed (`flash_reads` and `triton_sparse_read` in `src/lsa/sglang/models/qwen3_5/backend.py`).

## Results so far

The experiment stopped on 2026-11-04 after these runs, so the question above is only partly answered.
Both servers ran the recipes in this repo at commit f6cd164, on one GB10. Raw lines are in results/.

Speed at a 131,072-token prompt, one request at a time, 64 tokens out, two runs each:

| | Full attention | D-LSA | Gain |
|---|---:|---:|---:|
| Time to first token | 238 s | 131 s | 1.8x |
| Prefill | 550 tok/s | 1,000 tok/s | 1.8x |
| Decode | 3.64 tok/s | 4.43 tok/s | 1.2x |

The server log carried both `lsa-lsa: armed` and `lsa-lsa: indexing` under D-LSA and neither under full attention, so these runs show the plugin and the Triton read path working on sm121.

**Not measured:**

- **Accuracy under D-LSA.** `needles.py` ran on full attention only: 12 of 12 correct (10 at 32K, 2 at 128K). It never ran on D-LSA, so these numbers do not show that D-LSA answers as well as full attention.
- **256K and 512K.** Neither was run. The D-LSA server reported room for 541,042 tokens, but a 512K prompt has not been tried.

`needles.py` is a small check written for this recipe, not RULER, so its scores cannot be read against the paper.

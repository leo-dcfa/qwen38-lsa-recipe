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

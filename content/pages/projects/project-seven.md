---
type: ProjectLayout
title: MatShrink for Attention
colors: colors-a
date: '2026-04-24'
client: OpenMachine.ai
description: >-
  A lossless weight-compression transform for transformer attention, verified
  across all 270 layer/head pairs of SmolLM2-135M.
---
This is work done as a research collaborator with [OpenMachine.ai](https://openmachine.ai) on their open-source `transformer-tricks` library, contributing the proof-of-concept and precision study behind **MatShrink** applied to the V and O projections of attention.

The idea behind MatShrink is that attention has a redundancy hiding in plain sight. A head's value and output projections are only ever used as the product `V @ O` — nothing in the forward pass observes them separately. That means any invertible matrix `M` can be pushed through the pair without changing a single output:

```
V @ O  ==  (V @ M) @ (M⁻¹ @ O)
```

Choose `M` to be a `dk x dk` block of `O` itself, and `M⁻¹ @ O` has an identity block where that slice used to be. An identity block does not need to be stored. The weights get smaller, and the model's behavior is not approximated — it is *unchanged*.

### Deriving the transform

The transform is applied per KV group so that grouped-query attention is handled correctly: every query head sharing a KV head must receive the same `M`. The arithmetic is done in fp64 regardless of how the result is stored, because inverting a near-singular block in low precision is the one place this can genuinely go wrong:

```
def matshrink_vo(param, config, storage_dtype=torch.float16):
  """Apply V-O MatShrink to all layers of a Llama-family model.
     Math in fp64; weights written back in storage_dtype.
     Per KV group, pick M as first dk rows of the first query head's O slice."""
  h, h_kv = config.num_attention_heads, config.num_key_value_heads
  g, dk = h // h_kv, config.head_dim

  for layer in range(config.num_hidden_layers):
    # note: all weights are transposed in tensorfile
    Wv = param[tt.weight('V', layer)].to(torch.float64).numpy().T
    Wo = param[tt.weight('O', layer)].to(torch.float64).numpy().T

    for kv in range(h_kv):
      q0 = kv * g
      M = Wo[q0*dk:(q0+1)*dk, :dk].copy()
      M_inv = np.linalg.inv(M)
      Wv[:, kv*dk:(kv+1)*dk] = Wv[:, kv*dk:(kv+1)*dk] @ M
      for q in range(q0, q0 + g):
        Wo[q*dk:(q+1)*dk, :] = M_inv @ Wo[q*dk:(q+1)*dk, :]

    param[tt.weight('V', layer)] = torch.from_numpy(Wv.T).to(storage_dtype).contiguous()
    param[tt.weight('O', layer)] = torch.from_numpy(Wo.T).to(storage_dtype).contiguous()
```

### Verifying equivalence, not just accuracy

"Lossless" is a strong claim, and a benchmark score is a weak way to defend it — two different models can score the same. The stronger check is to compare the composition itself, head by head, before and after the transform. On SmolLM2-135M that is 30 layers x 9 heads = **270 pairs, all passing in fp64**:

```
for layer in range(config.num_hidden_layers):
  Wv0 = param_orig[tt.weight('V', layer)].to(torch.float64).numpy().T
  Wo0 = param_orig[tt.weight('O', layer)].to(torch.float64).numpy().T
  Wv1 = param_new [tt.weight('V', layer)].to(torch.float64).numpy().T
  Wo1 = param_new [tt.weight('O', layer)].to(torch.float64).numpy().T
  ok = all(np.allclose(Wv0[:, (q//g)*dk:(q//g+1)*dk] @ Wo0[q*dk:(q+1)*dk, :],
                       Wv1[:, (q//g)*dk:(q//g+1)*dk] @ Wo1[q*dk:(q+1)*dk, :])
           for q in range(h))
  print(layer, ':', ok)
```

### What storage precision costs

The transform is exact in fp64. The deployment question is what survives being written back at a precision people actually ship. Sweeping baseline against MatShrink perplexity at three storage dtypes:

| storage dtype | baseline | matshrink | diff |
| --- | --- | --- | --- |
| bf16 | 13.120 | 14.547 | +1.43 |
| fp16 | 13.112 | 13.124 | +0.012 |
| fp32 | 13.113 | 13.113 | 0.000 |

fp16 storage is effectively lossless at +0.09% perplexity drift. bf16 is not — it trades mantissa bits for exponent range, and the products produced by this transform need the mantissa. That distinction is the practical deployment guidance the study produced: **the transform is safe to ship in fp16, and should not be stored in bf16** even though bf16 is the more common training format.

Running the sweep at all was its own small problem — the perplexity helper was CPU- and dtype-fixed, so a separate patch added `device` and `dtype` arguments before the three-precision comparison could run in reasonable time.

### Merged contributions

*   [Add MatShrink V-O notebook](https://github.com/OpenMachine-ai/transformer-tricks/pull/11) — reproducible proof of concept, per-head equivalence check, three-precision perplexity sweep

*   [Add device and dtype args to perplexity](https://github.com/OpenMachine-ai/transformer-tricks/pull/10) — GPU support for the evaluation path

### For code:

<https://github.com/OpenMachine-ai/transformer-tricks>

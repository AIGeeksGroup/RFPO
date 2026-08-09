# H52 Surgical Output-Head FPO++ Audit

## Validity

The valid v4 control and candidate used the released `trc7rbt0_step_110000` EMA checkpoint, seed
`20260981`, 16 Square environments, two 320-step collections, identical MC8 settings, and ten actor
epochs at LR `1e-5`. Their collection-fingerprint JSON files are byte-identical. Both found 213
positive fully valid chunks and selected the same locked 64; pre-update ratios and surrogates match
exactly. Pre-update gradient norms differ by design because candidate gradients are represented only
in the output-head subspace.

The candidate selected exactly `model.mlp.6.weight` and `model.mlp.6.bias`: 114,800 parameters,
0.8587% of all actor parameters and 0.8839% of the control's non-visual trainable parameters. The
optimizer contained exactly those tensors. Both changed, while every frozen parameter remained
bitwise unchanged. All audited values were finite.

Two earlier launcher checks are invalid research runs and are excluded: v2 and v3 executed only the
critic-warmup iteration because the trainer's `total_timesteps` display and loop condition use
different units. An earlier v1 was interrupted at collection step 100 after the same budget issue
was noticed. None entered the iteration-2 actor audit or produced an audit JSON.

## Epoch-10 Results

| Batch | Metric | Control | Output head | Gate | Result |
|---:|---|---:|---:|---|---|
| 0 | gradient cosine | 0.16297 | 0.72179 | candidate >= control + 0.10 | pass (+0.55882) |
| 1 | gradient cosine | 0.34301 | 0.71577 | candidate >= control + 0.10 | pass (+0.37277) |
| 0 | surrogate gain | 0.044216 | 0.000923 | positive and >= 50% control | fail (2.09%) |
| 1 | surrogate gain | 0.025032 | 0.000564 | positive and >= 50% control | fail (2.25%) |
| 0 | zero-source endpoint drift MSE | 0.0030633 | 0.00003228 | <= 50% control | pass (1.05%) |
| 1 | zero-source endpoint drift MSE | 0.0027389 | 0.00002708 | <= 50% control | pass (0.99%) |

## Decision

H52 is refuted. Restricting FPO++ to the final velocity head strongly preserves gradient direction
and the deterministic endpoint, but it retains only about 2% of the useful control surrogate gain.
Its stability comes from making an almost inert reward update, not from a better adaptation tradeoff.
Per protocol, do not run a Square reward screen and do not try bias-only, the last two layers, LoRA,
a larger learning rate, another seed, or another checkpoint.

This OSMesa experiment is a paired mechanism screen, not an official EGL benchmark or real-robot
result.

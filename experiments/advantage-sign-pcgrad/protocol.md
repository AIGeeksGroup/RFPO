# Protocol: H32 Advantage-Sign Conflict Projection

## Hypothesis H32

For an on-policy FPO++ minibatch, decompose the official signed-GAE loss gradient into positive- and
negative-advantage branches. Keep the positive branch unchanged. If the negative branch conflicts
with it, remove only the negative branch's projection onto the positive branch, then sum the two.
This should prevent negative-GAE updates from cancelling useful positive-GAE progress while retaining
all orthogonal negative-action information.

The projection has no scalar loss coefficient. It is inactive whenever the two branches do not
conflict. This is distinct from positive-only optimization, reward reweighting, and BC-anchor PCGrad.

## Fixed Audit

- Checkpoint: exact released Can `95j3noe4_step_6000` EMA actor.
- Environment and renderer: Can under the isolated OSMesa stack.
- Seed: 20260902.
- Collection: one critic-only warmup iteration followed by a fresh iteration-2 Gaussian rollout,
  using 16 environments, 320 environment steps, 8 executed actions, 10 Euler steps, and official
  FPO++ settings.
- Select 64 fully valid chunks with observed terminal-return labels. Use a seeded selection and split
  them into two fixed 32-chunk batches. Each batch must contain at least eight positive and eight
  negative normalized-GAE chunks; stop as invalid if the data do not satisfy this support condition.
- Reuse exactly the stored MC8 CFM times and noises for every gradient in a batch.
- Control: official signed-GAE FPO++ gradient.
- Candidate: positive-GAE gradient plus the negative-GAE gradient after coefficient-free conflict
  projection against the positive branch.
- Reference: FPO++ gradient weighted by centered discounted returns to the observed terminal outcome,
  as in the prior critic and advantage audits. This reference is diagnostic only and is not used for
  training.

## Gates

All of the following must hold in both fixed batches:

1. Positive and negative gradient norms, control norm, candidate norm, and reference norm are finite
   and nonzero.
2. The positive/negative gradient dot product is negative, establishing non-vacuous conflict.
3. Candidate cosine to the outcome-reference gradient exceeds control by at least 0.10.
4. Candidate cosine to the outcome-reference gradient is at least 0.75.
5. Candidate cosine to the original signed control gradient is at least 0.90, preventing a wholesale
   replacement of the official update.
6. Candidate gradient norm is between 70% and 130% of the control norm.

Stop without optimizer integration, reward training, or projection variants if any gate fails. On a
full pass only, implement the same per-minibatch rule and run one matched short control/candidate
training screen before any confirmation. The later reward screen must use balanced per-environment
episode accounting; its exact seed and success gates will be committed separately before rollout.

# Protocol: Paired Parameter-Space ES Signal Audit on Square

## Hypothesis

Complete-episode Square success contains a reproducible mirrored finite-difference signal in a fixed
low-dimensional subspace of the released flow actor. If true, parameter-space ES can optimize the
actual benchmark objective without the critic, GAE, or CFM-ratio chain that failed in H0-H52.

## Locked Perturbation Family

- Initialization: released Square checkpoint `trc7rbt0_step_110000`, EMA weights.
- Actor behavior: Euler-10, 16 predicted/executed actions, deterministic zero source.
- Perturbed tensors: exactly the four MLP weight matrices `model.mlp.{0,2,4,6}.weight`. Vision,
  normalization buffers, biases, and every other parameter remain bitwise fixed.
- Fixed subspace: 16 independent Gaussian basis directions with seeds `20261000` through `20261015`.
- Normalize each tensor direction to its parameter L2 norm and set
  `||delta_l||_2 / ||theta_l||_2 = 0.01`; apply mirrored `theta +/- delta` policies.
- Do not tune the relative scale, perturb biases, change layer membership, add an adapter, increase
  direction count, or replace the basis after observing this audit.

## Locked Evaluation

- Replica A environment seeds: `20261020` through `20261023`.
- Replica B environment seeds: `20261024` through `20261027`.
- For each direction and sign, evaluate exactly one complete episode per seed. Reset the same seeds
  for the positive and negative policies (common random numbers), use balanced per-environment
  accounting, and record success, return, episode length, and the first predicted normalized action
  chunk. Total budget: `16 directions x 2 signs x 8 seeds = 256 episodes`.
- The policy is restored from an immutable in-memory anchor before every sign. Record exact tensor
  norms, parameter hashes/displacements, seeds, and per-episode outputs.
- This is an OSMesa paired method audit, not an official EGL benchmark or real-robot result.

## Validity Gates

All values must be finite. Every perturbed tensor must match the locked relative norm within `1e-5`;
unlisted parameters must remain bitwise unchanged; restoring the anchor must be bitwise exact. The
median across directions of the half positive/negative first-action RMS difference must lie in
`[0.01, 0.10]`, ruling out both behaviorally invisible and destructive perturbations. Positive and
negative evaluations must each contribute exactly eight episodes per direction.

## Signal Gates

Let `d_A` and `d_B` be the positive-minus-negative success-rate differences over the four seeds in
each replica. All gates must pass:

1. at least 6 of 16 directions have nonzero `d_A`, and at least 6 have nonzero `d_B`;
2. Pearson correlation between `d_A` and `d_B` is at least `0.35`;
3. at least five directions are nonzero in both replicas, with at least 70% sign agreement;
4. the four directions ranked highest by `d_A` have mean `d_B >= 0.125`, and the four ranked highest
   by `d_B` have mean `d_A >= 0.125`.

If any gate fails, refute H53 without scale, seed, basis, dimension, layer, or reward variants. If all
pass, separately commit a one-update fixed-subspace ES protocol with a matched base-policy control;
no multi-update training or official benchmark is authorized by this protocol.

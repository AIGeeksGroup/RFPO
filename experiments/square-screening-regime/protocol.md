# Protocol: Square Screening-Regime Validation

## Hypothesis

The released Square base checkpoint provides a non-saturated manipulation regime with enough
successes to compare short FPO++ continuations, avoiding Can step-1000 reward scarcity and
step-6000 screen saturation.

## Locked Evaluation

- Checkpoint: exact released `trc7rbt0_step_110000`, EMA weights
- Environment: `Square` under the existing isolated OSMesa runtime
- Seed: `20260912`
- Modes: official 10-step Euler inference with zero source and independent standard-Gaussian source
- Accounting: 20 environments, exactly one completed episode from each environment, 20 episodes per
  mode; no first-completion censoring
- No model, environment, action-step, integration, or source-scale changes
- OSMesa is accepted only to select a small experiment regime. Any final reward confirmation still
  requires healthy EGL.

## Gate

At least one source mode must record between 3 and 17 successes inclusive. Both modes must produce
20 completed finite episodes. If neither mode passes, do not start Square method training and move
to the next released benchmark without changing this seed or episode accounting.

If the gate passes, H40 is supported only as a screening-regime result. Separately register the
first Square method/control protocol before training; no method result may be inferred from H40.


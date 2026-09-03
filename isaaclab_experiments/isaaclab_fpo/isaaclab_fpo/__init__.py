"""IsaacLab extension for Flow Policy Optimization (FPO).

This package provides:
- FPO algorithm (flow-based policy optimization)
- Actor-critic modules with flow matching
- On-policy runner with EMA, post-training eval, multi-GPU support
- Config classes (runner, actor-critic, algorithm) with per-task defaults
- VecEnv wrapper for IsaacLab environments
- Policy exporters (JIT, ONNX)
"""


def __getattr__(name):
    if name in ("FpoRslRlOnPolicyRunnerCfg", "FpoRslRlPpoActorCriticCfg", "FpoRslRlPpoAlgorithmCfg"):
        from .rl_cfg import FpoRslRlOnPolicyRunnerCfg, FpoRslRlPpoActorCriticCfg, FpoRslRlPpoAlgorithmCfg
        return locals()[name]
    if name == "FpoRslRlVecEnvWrapper":
        from .wrapper import FpoRslRlVecEnvWrapper
        return FpoRslRlVecEnvWrapper
    if name in ("export_policy_as_jit", "export_policy_as_onnx"):
        from .exporter import export_policy_as_jit, export_policy_as_onnx
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

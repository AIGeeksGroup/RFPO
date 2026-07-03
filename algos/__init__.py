__all__ = ["ActorCriticPolicy", "DiagGaussianDistribution"]


def __getattr__(name):
    if name == "ActorCriticPolicy":
        from .policies import ActorCriticPolicy

        return ActorCriticPolicy
    if name == "DiagGaussianDistribution":
        from .distributions import DiagGaussianDistribution

        return DiagGaussianDistribution
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

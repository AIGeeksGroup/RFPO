import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parsed(path):
    return ast.parse((ROOT / path).read_text())


def test_robosuite_constructor_receives_wrapper_seed():
    tree = parsed("src/dexmg_env.py")
    wrapper = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "RobosuiteGymWrapper"
    )
    constructor = next(
        node
        for node in wrapper.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    seed_entries = [
        value
        for node in ast.walk(constructor)
        if isinstance(node, ast.Dict)
        for key, value in zip(node.keys, node.values)
        if isinstance(key, ast.Constant) and key.value == "seed"
    ]

    assert len(seed_entries) == 1
    assert isinstance(seed_entries[0], ast.Name)
    assert seed_entries[0].id == "seed"


def test_training_builds_distinct_reproducible_environment_seeds():
    tree = parsed("finetune_online_rl.py")
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "create_vectorized_env"
    ]
    training_call = next(
        call
        for call in calls
        if any(keyword.arg == "num_envs" for keyword in call.keywords)
    )
    seeds = next(keyword.value for keyword in training_call.keywords if keyword.arg == "seeds")

    assert isinstance(seeds, ast.ListComp)
    assert "cfg.seed + global_env_offset + env_id" in ast.unparse(seeds)


def test_evaluation_builds_distinct_reproducible_environment_seeds():
    tree = parsed("eval_checkpoint.py")
    evaluation_call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "create_vectorized_env"
    )
    seeds = next(
        keyword.value for keyword in evaluation_call.keywords if keyword.arg == "seeds"
    )

    assert isinstance(seeds, ast.IfExp)
    assert "cfg.seed is not None" in ast.unparse(seeds.test)
    assert "cfg.seed + env_id" in ast.unparse(seeds.body)
    assert isinstance(seeds.orelse, ast.Constant) and seeds.orelse.value is None

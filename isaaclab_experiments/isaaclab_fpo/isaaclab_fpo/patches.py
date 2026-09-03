"""Monkey-patches for upstream IsaacLab to support W&B sweep config overrides.

Patches applied to isaaclab.utils.dict.update_class_from_dict:
1. Allow list length changes for simple (non-dict) lists (e.g. hidden_dims)
2. int -> float casting tolerance (W&B sends ints for float params)
3. None member passthrough (allow setting values on None-initialized fields)
"""

from collections.abc import Iterable, Mapping


def _is_missing(obj):
    """Check if obj is a dataclasses MISSING sentinel (robust against module reloading)."""
    return type(obj).__name__ == "_MISSING_TYPE"


def apply_isaaclab_patches():
    """Apply monkey-patches to isaaclab.utils.dict at import time."""
    import isaaclab.utils.dict as dict_module
    from isaaclab.utils.string import callable_to_string, string_to_callable

    def update_class_from_dict(obj, data: dict, _ns: str = "") -> None:
        for key, value in data.items():
            key_ns = _ns + "/" + key
            if hasattr(obj, key) or isinstance(obj, dict):
                obj_mem = obj[key] if isinstance(obj, dict) else getattr(obj, key)
                if isinstance(value, Mapping):
                    update_class_from_dict(obj_mem, value, _ns=key_ns)
                    continue
                if isinstance(value, Iterable) and not isinstance(value, str):
                    contains_nested_dicts = obj_mem is not None and any(isinstance(v, dict) for v in value)
                    # Only enforce length matching for lists with nested dicts
                    if contains_nested_dicts and len(obj_mem) != len(value) and obj_mem is not None:
                        raise ValueError(
                            f"[Config]: Incorrect length under namespace: {key_ns}."
                            f" Expected: {len(obj_mem)}, Received: {len(value)}."
                        )
                    if isinstance(obj_mem, tuple):
                        value = tuple(value)
                    else:
                        set_obj = True
                        if contains_nested_dicts:
                            for i in range(len(obj_mem)):
                                if isinstance(value[i], dict):
                                    update_class_from_dict(obj_mem[i], value[i], _ns=key_ns)
                                    set_obj = False
                        if not set_obj:
                            continue
                elif callable(obj_mem):
                    value = string_to_callable(value)
                elif isinstance(value, float) and isinstance(obj_mem, float):
                    value = float(value)
                elif isinstance(value, int) and isinstance(obj_mem, float):
                    value = float(value)  # W&B sends ints for float params
                elif obj_mem is None or _is_missing(obj_mem):
                    value = value  # allow setting values on None/MISSING-initialized fields
                elif isinstance(value, type(obj_mem)) or value is None:
                    pass
                else:
                    raise ValueError(
                        f"[Config]: Incorrect type under namespace: {key_ns}."
                        f" Expected: {type(obj_mem)}, Received: {type(value)}."
                    )
                if isinstance(obj, dict):
                    obj[key] = value
                else:
                    setattr(obj, key, value)
            else:
                raise KeyError(f"[Config]: Key not found under namespace: {key_ns}.")

    dict_module.update_class_from_dict = update_class_from_dict

    # Also patch the direct import in configclass.py, which has:
    #   from .dict import update_class_from_dict
    # Without this, configclass.from_dict() would still use the unpatched version.
    import isaaclab.utils.configclass as configclass_module
    configclass_module.update_class_from_dict = update_class_from_dict

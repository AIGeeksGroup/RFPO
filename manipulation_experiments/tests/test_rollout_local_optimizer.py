import torch

from src.rollout_local_optimizer import clear_optimizer_state


def test_clear_optimizer_state_preserves_groups_and_parameters():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW(
        [parameter], lr=1e-5, betas=(0.9, 0.99), weight_decay=1e-6
    )
    (parameter.square().sum()).backward()
    optimizer.step()

    parameter_after_step = parameter.detach().clone()
    group_before = {
        key: value
        for key, value in optimizer.param_groups[0].items()
        if key != "params"
    }
    event = clear_optimizer_state(optimizer)

    assert event == {"state_entries_before": 1, "state_entries_after": 0}
    assert len(optimizer.state) == 0
    assert torch.equal(parameter, parameter_after_step)
    assert {
        key: value
        for key, value in optimizer.param_groups[0].items()
        if key != "params"
    } == group_before


def test_clear_optimizer_state_accepts_fresh_optimizer():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=1e-5)

    assert clear_optimizer_state(optimizer) == {
        "state_entries_before": 0,
        "state_entries_after": 0,
    }

from isaaclab_training.export.contracts import (get_export_contract, task_space_action_scale,
                                                TASK_SPACE_EXPORT_MODEL_NAME,
                                                task_space_policy_obs)
import pytest
import torch


def test_task_space_policy_obs_requires_policy_group():
    obs = {"critic": torch.zeros(1, 18)}
    with pytest.raises(KeyError):
        task_space_policy_obs(obs)


def test_task_space_policy_obs_requires_18_dimensions():
    obs = {"policy": torch.zeros(1, 17)}
    with pytest.raises(ValueError, match="18D"):
        task_space_policy_obs(obs)


def test_task_space_policy_obs_returns_policy_tensor():
    policy = torch.arange(18, dtype=torch.float32).reshape(1, 18)
    assert task_space_policy_obs({"policy": policy}) is policy


def test_task_space_action_scale_uses_env_cfg_action_scales():

    class _Action:
        position_scale = 0.025
        orientation_scale = 0.05

    class _Actions:
        arm_action = _Action()

    class _EnvCfg:
        actions = _Actions()

    scale = task_space_action_scale(_EnvCfg(), device="cpu", dtype=torch.float32)

    torch.testing.assert_close(scale, torch.tensor([[0.025, 0.025, 0.025, 0.05, 0.05, 0.05]]))
    assert TASK_SPACE_EXPORT_MODEL_NAME == "DisplayPortTaskSpace"


def test_contract_registry_resolves_displayport_task_space():
    contract = get_export_contract("displayport_task_space")
    assert contract is not None
    assert contract.name == "displayport_task_space"
    assert contract.skip_automatic_env_patch is True

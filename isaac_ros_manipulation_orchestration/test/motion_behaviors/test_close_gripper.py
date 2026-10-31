# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for resolving class-specific gripper close positions."""

from isaac_ros_manipulation_orchestration.behaviors.motion_behaviors import CloseGripper
from isaac_ros_manipulation_orchestration.utils.params_loader import CloseGripperConfig


DEFAULT_CLOSE_POSITION = 0.55
MAC_AND_CHEESE_CLOSE_POSITION = 0.56
SOUP_CAN_CLOSE_POSITION = 0.42


class FakeBlackboard:
    """Minimal blackboard interface used by ``CloseGripper`` resolution tests."""

    def __init__(self, **values):
        self._keys = set(values)
        for key, value in values.items():
            setattr(self, key, value)

    def exists(self, key):
        """Return whether a value was supplied for ``key``."""
        return key in self._keys


def make_close_gripper(close_positions_by_class=None):
    """Create a close behavior without setting up its ROS action client."""
    return CloseGripper(
        name='Close Gripper',
        gripper_action_name='/test/gripper_cmd',
        close_position=DEFAULT_CLOSE_POSITION,
        max_effort=10.0,
        close_positions_by_class=close_positions_by_class,
    )


def test_close_gripper_config_normalizes_class_keys_and_values():
    """YAML-style numeric keys and values are normalized for runtime lookup."""
    config = CloseGripperConfig.from_dict({
        'close_position': DEFAULT_CLOSE_POSITION,
        'close_positions_by_class': {
            22: MAC_AND_CHEESE_CLOSE_POSITION,
            '3': str(SOUP_CAN_CLOSE_POSITION),
        },
    })

    assert config.close_positions_by_class == {
        '22': MAC_AND_CHEESE_CLOSE_POSITION,
        '3': SOUP_CAN_CLOSE_POSITION,
    }


def test_close_gripper_config_defaults_to_empty_class_map():
    """Existing configurations without per-class positions remain valid."""
    config = CloseGripperConfig.from_dict({})

    assert config.close_position == DEFAULT_CLOSE_POSITION
    assert config.close_positions_by_class == {}


def test_close_gripper_uses_active_object_class_override():
    """The active object's class selects its configured closure target."""
    behavior = make_close_gripper({'22': MAC_AND_CHEESE_CLOSE_POSITION})
    behavior.blackboard = FakeBlackboard(
        active_obj_id=7,
        object_info_cache={7: {'class_id': 22}},
    )

    close_position, class_id = behavior._get_close_position()

    assert close_position == MAC_AND_CHEESE_CLOSE_POSITION
    assert class_id == '22'


def test_close_gripper_falls_back_for_unconfigured_class():
    """An unlisted class uses the backward-compatible fallback position."""
    behavior = make_close_gripper({'22': MAC_AND_CHEESE_CLOSE_POSITION})
    behavior.blackboard = FakeBlackboard(
        active_obj_id=8,
        object_info_cache={8: {'class_id': '3'}},
    )

    close_position, class_id = behavior._get_close_position()

    assert close_position == DEFAULT_CLOSE_POSITION
    assert class_id == '3'


def test_close_gripper_falls_back_without_active_object():
    """Missing object context uses the fallback instead of failing a close."""
    behavior = make_close_gripper({'22': MAC_AND_CHEESE_CLOSE_POSITION})
    behavior.blackboard = FakeBlackboard()

    close_position, class_id = behavior._get_close_position()

    assert close_position == DEFAULT_CLOSE_POSITION
    assert class_id is None

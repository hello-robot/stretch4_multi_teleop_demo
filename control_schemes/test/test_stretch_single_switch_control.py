#!/usr/bin/env python3
"""Tests for StretchSingleSwitchControlNode.handle_joy() / transition_state().

Only handle_joy() (and the state transition it triggers) is exercised -- never
timer_callback(), which is this node's command-loop equivalent and the only
place that actually publishes motion commands. ToolFrameKinematics() in the
constructor only loads a URDF model via Pinocchio; it never touches hardware.
"""

from unittest.mock import MagicMock

import pytest
from sensor_msgs.msg import JointState

from control_schemes.stretch_single_switch_control import StretchSingleSwitchControlNode


@pytest.fixture
def node():
    n = StretchSingleSwitchControlNode()
    yield n
    n.destroy_node()


def test_initial_state_is_stopped(node):
    assert node._state == 0


def test_rising_edge_advances_state(node):
    node.handle_joy([], [1])
    assert node._state == 1


def test_held_button_does_not_re_trigger(node):
    node.handle_joy([], [1])
    assert node._state == 1
    node.handle_joy([], [1])  # still held, no rising edge
    assert node._state == 1


def test_release_then_press_triggers_next_transition(node):
    node.handle_joy([], [1])
    assert node._state == 1
    node.handle_joy([], [0])  # release
    node.handle_joy([], [1])  # rising edge again
    assert node._state == 2


def test_full_state_cycle_wraps_to_stopped(node):
    for expected in [1, 2, 3, 4, 0]:
        node.handle_joy([], [0])
        node.handle_joy([], [1])
        assert node._state == expected


def test_state_1_sets_exploration_direction_positive(node):
    node.handle_joy([], [1])
    assert node._state == 1
    assert node._exploration_direction == 1


def test_state_4_captures_selected_velocity_from_exploration_direction(node):
    # Drive to state 4: 0->1->2->3->4
    for _ in range(4):
        node.handle_joy([], [0])
        node.handle_joy([], [1])
    assert node._state == 4
    max_linear = node.get_parameter("exploration_speed.linear").value
    assert node._selected_v_x == pytest.approx(node._exploration_direction * max_linear)


def test_empty_buttons_list_is_ignored(node):
    node.handle_joy([], [])
    assert node._state == 0


def test_state_0_still_recenters_gripper(node):
    """Assert state 0 (Stopped) still publishes the gripper-recentering pos_msg.

    Publishers are mocked so nothing real is contacted and the node is never
    spun -- this calls timer_callback() directly, per the safety rule allowing
    that specific call in this fixing pass.
    """
    node.pos_pub.publish = MagicMock()
    node.vel_pub.publish = MagicMock()
    node.base_pub.publish = MagicMock()

    msg = JointState()
    msg.name = [
        "lift_joint", "arm_l1_joint", "arm_l2_joint", "arm_l3_joint", "arm_l4_joint",
        "wrist_yaw_joint", "wrist_pitch_joint", "wrist_roll_joint",
    ]
    msg.position = [0.5, 0.05, 0.05, 0.05, 0.05, 0.0, 0.0, 0.0]
    node.joint_states_callback(msg)

    assert node._state == 0
    node.timer_callback()

    node.pos_pub.publish.assert_called_once()
    pos_msg = node.pos_pub.publish.call_args[0][0]
    assert pos_msg.name == ["stretch_gripper"]
    assert list(pos_msg.position) == [0.0]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

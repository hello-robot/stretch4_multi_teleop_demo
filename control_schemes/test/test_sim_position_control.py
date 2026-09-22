#!/usr/bin/env python3
"""Tests for SimDirectPositionControlNode.handle_joy().

Only handle_joy() is exercised -- never command_loop() (this node's
command-loop equivalent, which is the only thing that ever calls
sim.move_to()/sim.set_base_velocity()). Stretch4MujocoSimulator is patched
out at construction time so the MuJoCo sim is never actually started; the
node is never spun either, so its 20Hz cmd_timer never fires.
"""

from unittest.mock import MagicMock, patch

import pytest

from control_schemes.sim_position_control import SimDirectPositionControlNode


@pytest.fixture
def node():
    with patch("control_schemes.sim_position_control.Stretch4MujocoSimulator") as MockSim:
        MockSim.return_value = MagicMock()
        n = SimDirectPositionControlNode()
        yield n
        n.destroy_node()


AXES6 = [0.0] * 6
BUTTONS6 = [0] * 6


def test_wrong_size_message_is_ignored(node):
    node.handle_joy([0.0] * 3, [0] * 6)
    assert node.target_positions == {}


def test_axis_values_are_mapped_into_joint_ranges(node):
    axes = list(AXES6)
    axes[0] = 1.0  # Lift -> upper range
    axes[1] = -1.0  # Arm -> lower range
    node.handle_joy(axes, list(BUTTONS6))
    lift_low, lift_high = node.ranges["lift"]
    arm_low, arm_high = node.ranges["arm"]
    assert node.target_positions["lift"] == pytest.approx(lift_high)
    assert node.target_positions["arm"] == pytest.approx(arm_low)


def test_forward_button_press_sets_positive_v_x(node):
    buttons = list(BUTTONS6)
    buttons[0] = 1  # Forward
    node.handle_joy(list(AXES6), buttons)
    assert node.target_v_x == pytest.approx(1.0)


def test_forward_then_backward_switches_direction(node):
    buttons = list(BUTTONS6)
    buttons[0] = 1
    node.handle_joy(list(AXES6), buttons)
    assert node.target_v_x == pytest.approx(1.0)

    buttons = [0, 1, 0, 0, 0, 0]  # release forward, press backward
    node.handle_joy(list(AXES6), buttons)
    assert node.target_v_x == pytest.approx(-1.0)


def test_releasing_forward_without_backward_stops(node):
    buttons = list(BUTTONS6)
    buttons[0] = 1
    node.handle_joy(list(AXES6), buttons)
    assert node.target_v_x == pytest.approx(1.0)

    node.handle_joy(list(AXES6), list(BUTTONS6))  # both released
    assert node.target_v_x == pytest.approx(0.0)


def test_rotate_ccw_and_cw_buttons_set_omega(node):
    buttons = list(BUTTONS6)
    buttons[5] = 1  # rotate_ccw
    node.handle_joy(list(AXES6), buttons)
    assert node.target_omega == pytest.approx(1.5)

    buttons = [0, 0, 0, 0, 1, 0]  # rotate_cw
    node.handle_joy(list(AXES6), buttons)
    assert node.target_omega == pytest.approx(-1.5)


def test_left_right_buttons_set_v_y(node):
    buttons = list(BUTTONS6)
    buttons[2] = 1  # Left
    node.handle_joy(list(AXES6), buttons)
    assert node.target_v_y == pytest.approx(1.0)


def test_prev_buttons_state_is_tracked(node):
    buttons = [1, 0, 0, 0, 0, 0]
    node.handle_joy(list(AXES6), buttons)
    assert node.prev_buttons == buttons


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

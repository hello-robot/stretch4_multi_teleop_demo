#!/usr/bin/env python3
"""Tests for StretchControlNode.handle_joy() across position/velocity/mixed modes.

Only handle_joy() is exercised (pure state -> publish-call computation). The
node is never spun and command_loop-equivalent timers never fire, so nothing
is ever actually sent toward a driver or robot: publishers are swapped for
Mocks immediately after construction so no real DDS message is emitted.
"""

from unittest.mock import MagicMock

import pytest

from control_schemes.stretch_control_node import StretchControlNode


def make_node(mode):
    node = StretchControlNode(mode=mode)
    # Safety: never let a real publish reach the DDS layer during tests.
    node.pos_pub.publish = MagicMock()
    node.vel_pub.publish = MagicMock()
    node.base_pub.publish = MagicMock()
    return node


@pytest.fixture
def position_node():
    node = make_node("position")
    yield node
    node.destroy_node()


@pytest.fixture
def velocity_node():
    node = make_node("velocity")
    yield node
    node.destroy_node()


@pytest.fixture
def mixed_node():
    node = make_node("mixed")
    yield node
    node.destroy_node()


AXES9 = [0.0] * 9


def test_position_mode_always_reports_position_control(position_node):
    assert position_node.is_position_control_mode([]) is True


def test_velocity_mode_always_reports_velocity_control(velocity_node):
    assert velocity_node.is_position_control_mode([]) is False


def test_mixed_mode_button_selects_position_control(mixed_node):
    assert mixed_node.is_position_control_mode([1]) is True
    assert mixed_node.is_position_control_mode([0]) is False
    assert mixed_node.is_position_control_mode([]) is False


def test_map_range_maps_extremes_to_declared_limits(position_node):
    lower = position_node.get_parameter("limit.lift.lower").value
    upper = position_node.get_parameter("limit.lift.upper").value
    assert position_node.map_range(-1.0, "lift") == pytest.approx(lower)
    assert position_node.map_range(1.0, "lift") == pytest.approx(upper)
    assert position_node.map_range(0.0, "lift") == pytest.approx((lower + upper) / 2)


def test_position_mode_publishes_positions_and_zero_velocity(position_node):
    axes = list(AXES9)
    axes[0] = 1.0  # lift -> upper limit
    position_node.handle_joy(axes, [])

    position_node.pos_pub.publish.assert_called_once()
    position_node.vel_pub.publish.assert_called_once()
    position_node.base_pub.publish.assert_called_once()

    pos_msg = position_node.pos_pub.publish.call_args[0][0]
    upper = position_node.get_parameter("limit.lift.upper").value
    assert pos_msg.name == ["lift", "arm", "wrist_roll", "wrist_pitch", "wrist_yaw", "stretch_gripper"]
    assert pos_msg.position[0] == pytest.approx(upper)

    vel_msg = position_node.vel_pub.publish.call_args[0][0]
    assert list(vel_msg.velocity) == [0.0] * 6


def test_velocity_mode_publishes_scaled_velocities_with_sign_flips(velocity_node):
    axes = list(AXES9)
    axes[0] = 0.5  # lift
    axes[4] = 0.5  # arm
    axes[5] = 0.5  # wrist_roll (sign-flipped)
    axes[6] = 0.5  # wrist_pitch (sign-flipped)
    axes[7] = 0.5  # wrist_yaw (sign-flipped)
    axes[8] = 0.5  # gripper
    velocity_node.handle_joy(axes, [])

    velocity_node.pos_pub.publish.assert_not_called()
    vel_msg = velocity_node.vel_pub.publish.call_args[0][0]

    max_lift = velocity_node.get_parameter("max_joint_vel.lift").value
    max_arm = velocity_node.get_parameter("max_joint_vel.arm").value
    max_roll = velocity_node.get_parameter("max_joint_vel.wrist_roll").value
    max_pitch = velocity_node.get_parameter("max_joint_vel.wrist_pitch").value
    max_yaw = velocity_node.get_parameter("max_joint_vel.wrist_yaw").value
    max_grip = velocity_node.get_parameter("max_joint_vel.gripper").value

    assert vel_msg.velocity[0] == pytest.approx(0.5 * max_lift)
    assert vel_msg.velocity[1] == pytest.approx(0.5 * max_arm)
    assert vel_msg.velocity[2] == pytest.approx(-0.5 * max_roll)
    assert vel_msg.velocity[3] == pytest.approx(-0.5 * max_pitch)
    assert vel_msg.velocity[4] == pytest.approx(-0.5 * max_yaw)
    assert vel_msg.velocity[5] == pytest.approx(0.5 * max_grip)


def test_base_twist_always_published_from_axes_1_2_3(position_node):
    axes = list(AXES9)
    axes[1] = 1.0
    axes[2] = -1.0
    axes[3] = 0.5
    position_node.handle_joy(axes, [])

    twist = position_node.base_pub.publish.call_args[0][0]
    max_linear = position_node.get_parameter("max_linear_vel").value
    max_angular = position_node.get_parameter("max_angular_vel").value
    assert twist.linear.x == pytest.approx(1.0 * max_linear)
    assert twist.linear.y == pytest.approx(-1.0 * max_linear)
    assert twist.angular.z == pytest.approx(0.5 * max_angular)


def test_mixed_mode_switches_between_position_and_velocity_publish(mixed_node):
    axes = list(AXES9)
    mixed_node.handle_joy(axes, [1])  # Hold-for-position pressed
    assert mixed_node.pos_pub.publish.called
    mixed_node.pos_pub.publish.reset_mock()
    mixed_node.vel_pub.publish.reset_mock()

    mixed_node.handle_joy(axes, [0])  # released -> velocity mode
    assert not mixed_node.pos_pub.publish.called
    assert mixed_node.vel_pub.publish.called


def test_too_few_axes_is_ignored(position_node):
    position_node.handle_joy([0.0] * 3, [])
    position_node.pos_pub.publish.assert_not_called()
    position_node.vel_pub.publish.assert_not_called()
    position_node.base_pub.publish.assert_not_called()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

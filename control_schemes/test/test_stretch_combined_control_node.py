#!/usr/bin/env python3
"""Tests for StretchCombinedControlNode.handle_joy().

Only handle_joy() is exercised. Publishers are swapped for Mocks right after
construction so no real DDS message is ever emitted; the node is never spun.
"""

from unittest.mock import MagicMock

import pytest
from sensor_msgs.msg import JointState

from control_schemes.stretch_combined_control_node import StretchCombinedControlNode


@pytest.fixture
def node():
    n = StretchCombinedControlNode()
    n.vel_pub.publish = MagicMock()
    n.base_pub.publish = MagicMock()
    yield n
    n.destroy_node()


AXES7 = [0.0] * 7


def set_joint_state(node, **positions):
    msg = JointState()
    msg.name = list(positions.keys())
    msg.position = list(positions.values())
    node.joint_states_callback(msg)


def test_too_few_axes_is_ignored(node):
    node.handle_joy([0.0] * 3, [])
    node.vel_pub.publish.assert_not_called()
    node.base_pub.publish.assert_not_called()


def test_normal_mode_base_and_arm_and_roll(node):
    axes = list(AXES7)
    axes[2] = 0.5   # Base X
    axes[3] = -0.5  # Base Y
    axes[4] = 0.3   # Arm
    axes[5] = 0.4   # Wrist Roll
    axes[6] = 0.2   # Gripper
    node.handle_joy(axes, [])

    twist = node.base_pub.publish.call_args[0][0]
    max_linear = node.get_parameter("max_linear_vel").value
    max_arm = node.get_parameter("max_joint_vel.arm").value
    max_roll = node.get_parameter("max_joint_vel.wrist_roll").value
    max_grip = node.get_parameter("max_joint_vel.gripper").value

    assert twist.linear.x == pytest.approx(0.5 * max_linear)
    assert twist.linear.y == pytest.approx(-0.5 * max_linear)

    vel_msg = node.vel_pub.publish.call_args[0][0]
    # vel_msg.name = ["lift", "arm", "wrist_roll", "wrist_pitch", "wrist_yaw", "stretch_gripper"]
    assert vel_msg.velocity[1] == pytest.approx(0.3 * max_arm)
    assert vel_msg.velocity[2] == pytest.approx(-0.4 * max_roll)
    assert vel_msg.velocity[5] == pytest.approx(0.2 * max_grip)


def test_combined_pitch_lift_moves_wrist_before_split_limit(node):
    set_joint_state(node, wrist_pitch_joint=0.0, wrist_yaw_joint=0.0)
    axes = list(AXES7)
    axes[0] = 1.0  # Combined Pitch Lift, "up"
    node.handle_joy(axes, [])

    vel_msg = node.vel_pub.publish.call_args[0][0]
    max_pitch = node.get_parameter("max_joint_vel.wrist_pitch").value
    # Below the split limit: wrist_pitch moves, lift stays at 0.
    assert vel_msg.velocity[3] == pytest.approx(-1.0 * max_pitch)
    assert vel_msg.velocity[0] == pytest.approx(0.0)


def test_combined_pitch_lift_moves_lift_past_split_limit(node):
    # curr_pitch beyond -45deg split threshold -> switches to lift.
    set_joint_state(node, wrist_pitch_joint=-1.0, wrist_yaw_joint=0.0)
    axes = list(AXES7)
    axes[0] = 1.0
    node.handle_joy(axes, [])

    vel_msg = node.vel_pub.publish.call_args[0][0]
    max_lift = node.get_parameter("max_joint_vel.lift").value
    assert vel_msg.velocity[3] == pytest.approx(0.0)
    assert vel_msg.velocity[0] == pytest.approx(1.0 * max_lift)


def test_combined_yaw_turn_moves_wrist_before_split_limit(node):
    set_joint_state(node, wrist_pitch_joint=0.0, wrist_yaw_joint=0.0)
    axes = list(AXES7)
    axes[1] = 1.0  # Combined Yaw Turn
    node.handle_joy(axes, [])

    vel_msg = node.vel_pub.publish.call_args[0][0]
    twist = node.base_pub.publish.call_args[0][0]
    max_yaw = node.get_parameter("max_joint_vel.wrist_yaw").value
    assert vel_msg.velocity[4] == pytest.approx(-1.0 * max_yaw)
    assert twist.angular.z == pytest.approx(0.0)


def test_freeze_lift_base_button(node):
    set_joint_state(node, wrist_pitch_joint=0.0, wrist_yaw_joint=0.0)
    axes = list(AXES7)
    axes[0] = 1.0
    axes[1] = 1.0
    axes[2] = 1.0
    node.handle_joy(axes, [1, 0, 0])

    twist = node.base_pub.publish.call_args[0][0]
    vel_msg = node.vel_pub.publish.call_args[0][0]
    max_pitch = node.get_parameter("max_joint_vel.wrist_pitch").value
    max_yaw = node.get_parameter("max_joint_vel.wrist_yaw").value

    assert twist.linear.x == 0.0
    assert vel_msg.velocity[0] == 0.0  # lift frozen
    # Wrist moves directly without split limits.
    assert vel_msg.velocity[3] == pytest.approx(-1.0 * max_pitch)
    assert vel_msg.velocity[4] == pytest.approx(-1.0 * max_yaw)


def test_freeze_wrist_button(node):
    axes = list(AXES7)
    axes[0] = 1.0
    axes[1] = 1.0
    axes[2] = 0.5
    node.handle_joy(axes, [0, 1, 0])

    vel_msg = node.vel_pub.publish.call_args[0][0]
    twist = node.base_pub.publish.call_args[0][0]
    max_lift = node.get_parameter("max_joint_vel.lift").value
    max_angular = node.get_parameter("max_angular_vel").value

    assert vel_msg.velocity[3] == 0.0  # wrist_pitch frozen
    assert vel_msg.velocity[4] == 0.0  # wrist_yaw frozen
    assert vel_msg.velocity[0] == pytest.approx(1.0 * max_lift)
    assert twist.angular.z == pytest.approx(-1.0 * max_angular)


def test_both_freeze_buttons_is_safe_stop(node):
    axes = list(AXES7)
    axes[0] = 1.0
    axes[1] = 1.0
    axes[2] = 1.0
    axes[3] = 1.0
    node.handle_joy(axes, [1, 1, 0])

    twist = node.base_pub.publish.call_args[0][0]
    vel_msg = node.vel_pub.publish.call_args[0][0]
    assert twist.linear.x == 0.0
    assert twist.linear.y == 0.0
    assert twist.angular.z == 0.0
    assert vel_msg.velocity[0] == 0.0
    assert vel_msg.velocity[3] == 0.0
    assert vel_msg.velocity[4] == 0.0


def test_hold_to_go_home_drives_p_control_toward_targets(node):
    set_joint_state(
        node,
        lift_joint=0.7,
        arm_l1_joint=0.05, arm_l2_joint=0.05, arm_l3_joint=0.05, arm_l4_joint=0.05,
        wrist_roll_joint=0.1,
        wrist_pitch_joint=0.1,
        wrist_yaw_joint=0.1,
    )
    axes = list(AXES7)
    node.handle_joy(axes, [0, 0, 1])

    twist = node.base_pub.publish.call_args[0][0]
    vel_msg = node.vel_pub.publish.call_args[0][0]
    assert twist.linear.x == 0.0 and twist.linear.y == 0.0 and twist.angular.z == 0.0
    # p_lift=0.7 > target 0.5 -> negative lift velocity, clamped to -0.05.
    assert vel_msg.velocity[0] == pytest.approx(-0.05)
    # Fixed behavior: the P-control arm/wrist_roll homing values computed in the
    # hold_to_go_home branch are no longer clobbered by the "Arm and Wrist Roll always
    # move normally" block (that block now only runs in normal mode), so arm and
    # wrist_roll do actually drive toward their home targets.
    # p_arm = 4*0.05 = 0.2 > target 0.15 -> negative arm velocity, clamped to -0.05.
    assert vel_msg.velocity[1] == pytest.approx(-0.05)
    # p_roll=0.1 > target 0.0 -> roll: 0.0 - 0.1, within +/-0.20 clamp.
    assert vel_msg.velocity[2] == pytest.approx(-0.1)
    assert vel_msg.velocity[3] == pytest.approx(-0.1)  # pitch: 0.0 - 0.1, within +/-0.20 clamp
    assert vel_msg.velocity[4] == pytest.approx(-0.1)  # yaw: 0.0 - 0.1, within +/-0.20 clamp


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

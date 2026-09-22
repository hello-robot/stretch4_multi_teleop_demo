#!/usr/bin/env python3
"""Tests for StretchKinematicControlNode.handle_joy().

Only handle_joy() is exercised. Publishers are swapped for Mocks right after
construction so no real DDS message is ever emitted; the node is never spun.
tf_buffer.lookup_transform is patched per-test to avoid depending on a live
/tf stream (this node's constructor never touches hardware on its own --
ToolFrameKinematics() only loads a URDF, and TransformListener only
subscribes/listens).
"""

from unittest.mock import MagicMock

import pytest
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped
from tf2_ros import LookupException

from control_schemes.stretch_kinematic_control import StretchKinematicControlNode


def identity_transform():
    t = TransformStamped()
    t.transform.rotation.x = 0.0
    t.transform.rotation.y = 0.0
    t.transform.rotation.z = 0.0
    t.transform.rotation.w = 1.0
    return t


@pytest.fixture
def node():
    n = StretchKinematicControlNode()
    n.vel_pub.publish = MagicMock()
    n.base_pub.publish = MagicMock()
    n.tf_buffer.lookup_transform = MagicMock(return_value=identity_transform())
    yield n
    n.destroy_node()


def set_joint_state(node):
    msg = JointState()
    msg.name = [
        "lift_joint", "arm_l1_joint", "arm_l2_joint", "arm_l3_joint", "arm_l4_joint",
        "wrist_yaw_joint", "wrist_pitch_joint", "wrist_roll_joint",
    ]
    msg.position = [0.5, 0.05, 0.05, 0.05, 0.05, 0.0, 0.0, 0.0]
    node._joint_states_callback(msg)


AXES7 = [0.0] * 7


def test_too_few_axes_is_ignored(node):
    node.handle_joy([0.0] * 3, [])
    node.vel_pub.publish.assert_not_called()
    node.base_pub.publish.assert_not_called()


def test_no_joint_states_yet_is_ignored(node):
    node.handle_joy(list(AXES7), [0])
    node.vel_pub.publish.assert_not_called()
    node.base_pub.publish.assert_not_called()


def test_tf_lookup_failure_is_ignored(node):
    set_joint_state(node)
    node.tf_buffer.lookup_transform.side_effect = LookupException("no transform")
    node.handle_joy(list(AXES7), [0])
    node.vel_pub.publish.assert_not_called()
    node.base_pub.publish.assert_not_called()


def test_zero_axes_still_publishes_zero_velocity_for_first_few_cycles(node):
    set_joint_state(node)
    node.handle_joy(list(AXES7), [0])
    node.vel_pub.publish.assert_called_once()
    node.base_pub.publish.assert_called_once()
    vel_msg = node.vel_pub.publish.call_args[0][0]
    assert all(v == 0.0 for v in vel_msg.velocity)


def test_zero_velocity_stops_publishing_after_repeat_count(node):
    set_joint_state(node)
    repeat_count = node.get_parameter("zero_velocity_repeat_count").value
    for _ in range(repeat_count):
        node.handle_joy(list(AXES7), [0])
    call_count_before = node.vel_pub.publish.call_count
    node.handle_joy(list(AXES7), [0])  # one more all-zero cycle: should be suppressed
    assert node.vel_pub.publish.call_count == call_count_before


def test_nonzero_gripper_axis_publishes_gripper_velocity(node):
    set_joint_state(node)
    axes = list(AXES7)
    axes[6] = 1.0  # Gripper
    node.handle_joy(axes, [0])
    vel_msg = node.vel_pub.publish.call_args[0][0]
    max_grip = node.get_parameter("max_joint_vel.gripper").value
    assert vel_msg.name[-1] == "stretch_gripper"
    assert vel_msg.velocity[-1] == pytest.approx(1.0 * max_grip)


def test_button_press_switches_active_reference_frame(node):
    set_joint_state(node)
    assert node._active_frame == node._target_frame
    n_frames = len(node._reference_frames)
    buttons = [0] * n_frames
    if n_frames > 1:
        buttons[1] = 1
        node.handle_joy(list(AXES7), buttons)
        assert node._active_frame == node._reference_frames[1]


def test_active_frame_persists_without_new_press(node):
    set_joint_state(node)
    n_frames = len(node._reference_frames)
    if n_frames > 1:
        buttons = [0] * n_frames
        buttons[1] = 1
        node.handle_joy(list(AXES7), buttons)
        assert node._active_frame == node._reference_frames[1]
        buttons[1] = 0
        node.handle_joy(list(AXES7), buttons)
        assert node._active_frame == node._reference_frames[1]


def test_positive_x_axis_produces_forward_base_or_arm_motion(node):
    set_joint_state(node)
    axes = list(AXES7)
    axes[0] = 1.0  # X translation in target-frame-local coordinates
    node.handle_joy(axes, [0])
    node.vel_pub.publish.assert_called_once()
    node.base_pub.publish.assert_called_once()
    vel_msg = node.vel_pub.publish.call_args[0][0]
    twist = node.base_pub.publish.call_args[0][0]
    # Identity-rotation reference frame: v_desired local X is unchanged, so
    # *something* in the arm/lift/base should move; verify not everything is zero.
    moved = any(v != 0.0 for v in vel_msg.velocity) or twist.linear.x != 0.0
    assert moved


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

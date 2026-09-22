"""Pure-logic tests for MediaPipeBaseNode.add_virtual_item (no rclpy node needed).

Builds a MediaPipeBaseNode via object.__new__ with just the attributes
add_virtual_item touches, plus mocked get_logger()/set_parameters() to
observe whether the declared 'axis_names'/'button_names' ROS parameters get
synced, without needing a full ROS node/webcam.
"""
from unittest.mock import MagicMock

import pytest

from teleop_interfaces.mediapipe_base import MediaPipeBaseNode


class _ConcreteMediaPipeBase(MediaPipeBaseNode):
    """Minimal concrete subclass so object.__new__ can instantiate it (process_frame is abstract)."""

    def process_frame(self, frame):
        return None


def _make_base(ros_ready):
    node = object.__new__(_ConcreteMediaPipeBase)
    node._axes = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']
    node._buttons = []
    node._last_axes = [0.0] * len(node._axes)
    node._last_buttons = []
    node._virtual_config = []
    node._ros_ready = ros_ready
    node.set_parameters = MagicMock()
    node.get_logger = MagicMock(return_value=MagicMock())
    return node


def test_add_virtual_axis_item_appends_state():
    node = _make_base(ros_ready=True)
    node.add_virtual_item({'name': 'pinch', 'type': 'axis', 'p1': 'a', 'p2': 'b', 'max_dist': 1.0})
    assert node._axes[-1] == 'pinch'
    assert node._last_axes[-1] == 0.0
    assert node._virtual_config[-1]['name'] == 'pinch'


def test_add_virtual_button_item_appends_state():
    node = _make_base(ros_ready=True)
    node.add_virtual_item({'name': 'clench', 'type': 'button', 'p1': 'a', 'p2': 'b', 'threshold': 0.1})
    assert node._buttons[-1] == 'clench'
    assert node._last_buttons[-1] == 0


def test_add_virtual_item_before_ros_ready_does_not_touch_parameters():
    """During __init__'s initial config load (before super().__init__()), no Node exists yet."""
    node = _make_base(ros_ready=False)
    node.add_virtual_item({'name': 'pinch', 'type': 'axis', 'p1': 'a', 'p2': 'b'})
    assert 'pinch' in node._axes
    node.set_parameters.assert_not_called()


def test_add_virtual_item_after_ros_ready_syncs_axis_names_param():
    """A runtime-added (post-construction) virtual axis updates the declared parameter."""
    node = _make_base(ros_ready=True)
    node.add_virtual_item({'name': 'pinch', 'type': 'axis', 'p1': 'a', 'p2': 'b'})
    assert node.set_parameters.call_count == 1
    (params,), _ = node.set_parameters.call_args
    assert params[0].name == 'axis_names'
    assert list(params[0].value) == node._axes


def test_add_virtual_item_after_ros_ready_syncs_button_names_param():
    node = _make_base(ros_ready=True)
    node.add_virtual_item({'name': 'clench', 'type': 'button', 'p1': 'a', 'p2': 'b'})
    assert node.set_parameters.call_count == 1
    (params,), _ = node.set_parameters.call_args
    assert params[0].name == 'button_names'
    assert list(params[0].value) == node._buttons


def test_add_virtual_item_duplicate_name_is_ignored():
    node = _make_base(ros_ready=True)
    node.add_virtual_item({'name': 'pinch', 'type': 'axis', 'p1': 'a', 'p2': 'b'})
    node.set_parameters.reset_mock()
    node.add_virtual_item({'name': 'pinch', 'type': 'axis', 'p1': 'a', 'p2': 'b'})
    assert node._axes.count('pinch') == 1
    node.set_parameters.assert_not_called()


def test_add_virtual_item_unrecognized_type_warns_instead_of_silently_dropping():
    node = _make_base(ros_ready=True)
    node.add_virtual_item({'name': 'mystery', 'type': 'bogus', 'p1': 'a', 'p2': 'b'})
    assert 'mystery' not in node._axes
    assert 'mystery' not in node._buttons
    node.get_logger.return_value.warn.assert_called_once()


def test_add_virtual_item_missing_name_is_a_silent_noop():
    """No 'name' key: nothing to register, matches prior (harmless) behavior."""
    node = _make_base(ros_ready=True)
    node.add_virtual_item({'type': 'axis', 'p1': 'a', 'p2': 'b'})
    assert node._axes == ['x', 'y', 'z', 'roll', 'pitch', 'yaw']
    node.set_parameters.assert_not_called()

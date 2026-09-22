"""Mocked-hardware construction/smoke tests for gamepad_node.GamepadNode (evdev mocked out)."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import rclpy

from teleop_interfaces import gamepad_node
from teleop_interfaces.gamepad_node import GamepadNode


@pytest.fixture(scope='module', autouse=True)
def _rclpy_context():
    """Init/shutdown rclpy once for this test module."""
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def node():
    """Construct a GamepadNode with evdev device discovery disabled (no real hardware)."""
    with patch.object(gamepad_node.evdev, 'list_devices', return_value=[]):
        n = GamepadNode()
        yield n
        n.stop()
        n.destroy_node()


def test_constructs_and_publishes_on_expected_topic(node):
    """Node builds cleanly and its output publisher targets /gamepad/output."""
    assert node.output_publisher.topic_name == '/gamepad/output'
    assert node.get_name() == 'gamepad'


def test_no_device_found_leaves_device_none(node):
    """With no evdev devices present, the node has no connected device (no crash)."""
    assert node.device is None


def test_button_press_and_release_update_state(node):
    """EV_KEY events for a mapped button (BTN_A) set/clear the corresponding button index."""
    press = SimpleNamespace(type=gamepad_node.ecodes.EV_KEY, code=304, value=1)
    release = SimpleNamespace(type=gamepad_node.ecodes.EV_KEY, code=304, value=0)
    node._process_event(press)
    assert node._current_buttons[0] == 1
    node._process_event(release)
    assert node._current_buttons[0] == 0


def test_unmapped_key_code_is_ignored(node):
    """An EV_KEY event with a code not in BUTTON_MAP does not raise or change state."""
    before = list(node._current_buttons)
    event = SimpleNamespace(type=gamepad_node.ecodes.EV_KEY, code=999999, value=1)
    node._process_event(event)  # should not raise
    assert node._current_buttons == before


def test_stick_axis_deadzone_zeros_small_values(node):
    """A stick value within the deadzone parameter is reported as exactly 0.0."""
    deadzone = node.get_parameter('deadzone').value
    small_raw = int(deadzone * 32768.0 * 0.5)  # well inside the deadzone
    event = SimpleNamespace(type=gamepad_node.ecodes.EV_ABS, code=0, value=small_raw)  # ABS_X
    node._process_event(event)
    assert node._current_axes[0] == 0.0


def test_stick_axis_rescales_outside_deadzone(node):
    """A stick value outside the deadzone is rescaled into [-1, 1] and clamped."""
    event = SimpleNamespace(type=gamepad_node.ecodes.EV_ABS, code=0, value=32768)  # ABS_X, max
    node._process_event(event)
    assert node._current_axes[0] == pytest.approx(1.0, abs=1e-6)


def test_left_stick_y_axis_is_inverted(node):
    """ABS_Y (Left Stick Y) is inverted per AXIS_MAP, so a positive raw value goes negative."""
    event = SimpleNamespace(type=gamepad_node.ecodes.EV_ABS, code=1, value=32768)  # ABS_Y
    node._process_event(event)
    assert node._current_axes[1] < 0.0


def test_update_and_publish_does_not_raise(node):
    """update_and_publish (the timer callback) runs without error given current state."""
    node.update_and_publish()  # exercises publish_input via the base class

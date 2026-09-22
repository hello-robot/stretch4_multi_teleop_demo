"""Mocked-hardware construction/smoke tests for space_mouse_node.SpaceMouseNode."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import rclpy

from teleop_interfaces import space_mouse_node
from teleop_interfaces.space_mouse_node import SpaceMouseNode


@pytest.fixture(scope='module', autouse=True)
def _rclpy_context():
    """Init/shutdown rclpy once for this test module."""
    rclpy.init()
    yield
    rclpy.shutdown()


def _fake_state(x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0, buttons=(0, 0)):
    return SimpleNamespace(x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw, buttons=list(buttons))


@pytest.fixture
def node():
    """Construct a SpaceMouseNode with pyspacemouse.open() mocked out."""
    fake_device = MagicMock()
    fake_device.read.return_value = _fake_state()
    with patch.object(space_mouse_node.pyspacemouse, 'open', return_value=fake_device):
        n = SpaceMouseNode()
        yield n, fake_device
        n.stop()
        n.destroy_node()


def test_constructs_and_publishes_on_expected_topic(node):
    """Node builds cleanly and its output publisher targets /space_mouse/output."""
    n, _ = node
    assert n.output_publisher.topic_name == '/space_mouse/output'


def test_open_failure_reraises():
    """If pyspacemouse.open() raises, construction propagates the exception (current behavior)."""
    with patch.object(space_mouse_node.pyspacemouse, 'open',
                       side_effect=RuntimeError('no device')):
        with pytest.raises(RuntimeError):
            SpaceMouseNode()


def test_update_and_publish_clamps_axes(node):
    """Axis values outside [-1, 1] returned by the device are clamped before publishing."""
    n, fake_device = node
    fake_device.read.return_value = _fake_state(x=5.0, y=-5.0, buttons=(0, 0))
    n.update_and_publish()
    assert n._last_axes[0] == 1.0
    assert n._last_axes[1] == -1.0


def test_update_and_publish_pads_short_button_list(node):
    """A device state with fewer than 2 buttons is padded with zeros (current behavior)."""
    n, fake_device = node
    fake_device.read.return_value = _fake_state(buttons=(1,))
    n.update_and_publish()
    assert n._last_buttons == [1, 0]


def test_update_and_publish_truncates_long_button_list(node):
    """A device state with more than 2 buttons is truncated to 2 (current behavior)."""
    n, fake_device = node
    fake_device.read.return_value = _fake_state(buttons=(1, 1, 1))
    n.update_and_publish()
    assert n._last_buttons == [1, 1]


def test_update_and_publish_handles_none_state(node):
    """A None read() result (no data ready) is a no-op, not a crash."""
    n, fake_device = node
    fake_device.read.return_value = None
    n.update_and_publish()  # should not raise


def test_update_and_publish_swallows_read_exceptions(node):
    """An exception from device.read() is caught and logged, not propagated."""
    n, fake_device = node
    fake_device.read.side_effect = RuntimeError('disconnected')
    n.update_and_publish()  # should not raise

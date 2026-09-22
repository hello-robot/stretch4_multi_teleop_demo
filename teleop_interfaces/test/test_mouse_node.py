"""Mocked-hardware construction/smoke tests for mouse_node.MouseNode (pynput/tkinter mocked)."""
from unittest.mock import MagicMock, patch

import pytest
import rclpy

from teleop_interfaces import mouse_node
from teleop_interfaces.mouse_node import MouseNode


@pytest.fixture(scope='module', autouse=True)
def _rclpy_context():
    """Init/shutdown rclpy once for this test module."""
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def node():
    """Construct a MouseNode with pynput's listener and tkinter's screen probe mocked out."""
    fake_root = MagicMock()
    fake_root.winfo_screenwidth.return_value = 1000
    fake_root.winfo_screenheight.return_value = 500
    with patch.object(mouse_node.tk, 'Tk', return_value=fake_root), \
            patch.object(mouse_node.mouse, 'Listener') as fake_listener_cls:
        fake_listener_cls.return_value = MagicMock()
        n = MouseNode()
        yield n
        n.stop()
        n.destroy_node()


def test_constructs_and_publishes_on_expected_topic(node):
    """Node builds cleanly and its output publisher targets /mouse/output."""
    assert node.output_publisher.topic_name == '/mouse/output'
    assert node.get_parameter('screen_width').value == 1000
    assert node.get_parameter('screen_height').value == 500


def test_on_move_normalizes_to_range(node):
    """Cursor position is normalized to [-1, 1] using the detected screen size."""
    node._on_move(0, 0)
    assert node._current_x == pytest.approx(-1.0)
    assert node._current_y == pytest.approx(-1.0)
    node._on_move(1000, 500)
    assert node._current_x == pytest.approx(1.0)
    assert node._current_y == pytest.approx(1.0)
    node._on_move(500, 250)
    assert node._current_x == pytest.approx(0.0)
    assert node._current_y == pytest.approx(0.0)


def test_on_move_clamps_out_of_bounds_position(node):
    """A cursor position beyond the detected screen bounds is clamped to [-1, 1]."""
    node._on_move(5000, -5000)
    assert node._current_x == 1.0
    assert node._current_y == -1.0


def test_on_click_tracks_left_and_right_buttons(node):
    """Left/right press and release update the corresponding button flags independently."""
    node._on_click(0, 0, mouse_node.mouse.Button.left, True)
    assert node._left_pressed == 1
    node._on_click(0, 0, mouse_node.mouse.Button.left, False)
    assert node._left_pressed == 0
    node._on_click(0, 0, mouse_node.mouse.Button.right, True)
    assert node._right_pressed == 1


def test_on_scroll_accumulates_without_decay(node):
    """Scroll value accumulates across calls and is not reset between them (current behavior)."""
    node._on_scroll(0, 0, 0, 1)  # scroll up
    first = node._scroll_value
    assert first > 0.0
    node._on_scroll(0, 0, 0, 1)
    assert node._scroll_value > first  # keeps accumulating, no decay toward 0


def test_on_scroll_clamps_to_range(node):
    """Repeated scrolling in one direction clamps the accumulator to [-1, 1]."""
    for _ in range(1000):
        node._on_scroll(0, 0, 0, 1)
    assert node._scroll_value == 1.0
    for _ in range(1000):
        node._on_scroll(0, 0, 0, -1)
    assert node._scroll_value == -1.0


def test_update_and_publish_does_not_raise(node):
    """update_and_publish (the timer callback) runs without error given current state."""
    node.update_and_publish()
